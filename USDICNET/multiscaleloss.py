import torch
import torch.nn.functional as F


def EPE(input_flow, target_flow, rank, real=False, mean=True):
    """
        Args:
            input_flow: the predicted displacement
            target_flow : the real displacement
            rank : the device ID
            real : if false, gauss weight was used; else the endpoint error was calculated directly.
            mean : if true,the average endpoint error for a single point was calculated.

        Returns:
            endpoint error
    """
    if ~real:
        b, c, h, w = input_flow.size()

        X = torch.arange(-int(w / 2), int(w / 2)).to(rank)
        Y = torch.arange(-int(h / 2), int(h / 2)).to(rank)

        # gauss weight
        [x, y] = torch.meshgrid(X, Y)
        gauss_weight = torch.exp(-(x ** 2 + y ** 2) / (h * w / 8))

        # gauss_weight = torch.from_numpy(gauss_weight).float()
        # gauss_weight = gauss_weight
        # gauss_weight = (input_flow - target_flow) / torch.max(torch.abs(input_flow - target_flow))

        # weighted endpoint error
        EPE_map = torch.norm(input_flow.mul(gauss_weight) - target_flow.mul(gauss_weight), p=2, dim=1)  # 二阶范数，差值的平方和
        # EPE_map = torch.norm(input_flow - target_flow, 2, 1)
    else:
        EPE_map = torch.norm(input_flow - target_flow, p=2, dim=1)
    batch_size = EPE_map.size(0)

    if mean:
        # return pow(pow(EPE_map, 2).mean(), 0.5)
        return EPE_map.mean()/2
        #    math.sqrt(pow(EPE_map, 2).sum() / batch_size/np.prod(EPE_map.size()))
    else:
        # return pow(pow(EPE_map, 2).sum(), 0.5) / batch_size  # 矩阵范数，结果是输出与实际值差的平方和
        return EPE_map.sum() / batch_size


def multiscaleEPE(network_output, target_flow, rank, weights=None):
    """
        Args:
            network_output: the output of the network model
            target_flow : the real displacement
            rank : the device ID
            weights : the weight parameters for multi-scale predicted displacement to form the loss function

        Returns:
            loss :
    """

    def one_scale(output, target, rank):
        """
            Args:
                output: the output of the network model for a specific scale
                target : the real displacement
                rank : the device ID

            Returns:
                loss :
        """

        b, c, h, w = output.size()

        # obtain multi-scale target displacement field
        target_scaled = torch.nn.functional.interpolate(target, (h, w), mode='area')  # 根据图像尺寸实现插值和上采样

        return EPE(output, target_scaled, rank, real=False, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        # the defalut weight for multi-scale loss
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article 不同层的尺度分辨率不同，权重也不同
    assert (len(weights) == len(network_output))

    loss = 0
    for predicted_flow, weight in zip(network_output, weights):
        loss += weight * one_scale(predicted_flow, target_flow, rank)
    return loss


def realEPE(output, target, rank):
    """
        Args:
            output: the predicted displacement
            target : the real displacement
            rank : the device ID

        Returns:
            endpoint error
    """

    b, c, h, w = target.size()
    # sub_output = output[:, :, int(w / 4):int(w * 3 / 4), int(h / 4):int(h * 3 / 4)]
    # sub_target = target[:, :, int(w / 4):int(w * 3 / 4), int(h / 4):int(h * 3 / 4)]

    # evaluate the error of the predicted result without the influence of the image edge
    sub_output = output[:, :, 8:w - 8, 8:h - 8]
    # sub_target = target[:, :, 8:w - 8, 8:h - 8]
    sub_target = target[:, :, 8:w - 8, 8:h - 8]
    return EPE(sub_output, sub_target, rank, real=True, mean=True)


def loss_smooth(input1, input2, output, rou):
    _, _, h, w = output.size()
    dispx = output[0, 0, :, :]
    dispy = output[0, 1, :, :]
    loss_s = 0
    dispx_x = dispx.clone()
    dispy_x = dispy.clone()
    dispx_x[:, 1:w] = dispx[:, 0:w - 1]
    dispy_x[:, 1:w] = dispy[:, 0:w - 1]
    dispx_y = dispx.clone()
    dispy_y = dispy.clone()
    dispx_y[1:w, :] = dispx[0:w - 1, :]
    dispy_y[1:w, :] = dispy[0:w - 1, :]
    loss_s = F.mse_loss(dispx, dispx_x) + F.mse_loss(dispx, dispx_y) + \
             F.mse_loss(dispy, dispy_x) + F.mse_loss(dispy, dispy_y)
    # for x in range(0, w - 1):
    #     for y in range(0, h - 1):
    #         loss_s = loss_s + ((dispx[y, x] - dispx[y, x + 1]) ** 2 + (dispx[y, x] - dispx[y + 1, x]) ** 2 +
    #                            (dispy[y, x] - dispy[y, x + 1]) ** 2 + (dispy[y, x] - dispy[y + 1, x]) ** 2)
    # loss_s = (loss_s / (w * h)) ** 0.5
    # loss = F.mse_loss(input1, input2) + rou * loss_s
    return F.mse_loss(input1, input2), loss_s

def loss_smooth_disp_1(input1, input2, output, rou):
    _, _, h, w = output.size()
    dispx = output[0, 0, :, :]
    dispy = output[0, 1, :, :]
    loss_s = 0
    dispx_x = dispx.clone()
    dispy_x = dispy.clone()
    dispx_x[:, 1:w] = dispx[:, 0:w - 1]
    dispy_x[:, 1:w] = dispy[:, 0:w - 1]
    dispx_y = dispx.clone()
    dispy_y = dispy.clone()
    dispx_y[1:w, :] = dispx[0:w - 1, :]
    dispy_y[1:w, :] = dispy[0:w - 1, :]
    loss_s = F.mse_loss(dispx, dispx_x) + F.mse_loss(dispx, dispx_y) + \
             F.mse_loss(dispy, dispy_x) + F.mse_loss(dispy, dispy_y)
    # for x in range(0, w - 1):
    #     for y in range(0, h - 1):
    #         loss_s = loss_s + ((dispx[y, x] - dispx[y, x + 1]) ** 2 + (dispx[y, x] - dispx[y + 1, x]) ** 2 +
    #                            (dispy[y, x] - dispy[y, x + 1]) ** 2 + (dispy[y, x] - dispy[y + 1, x]) ** 2)
    # loss_s = (loss_s / (w * h)) ** 0.5
    # loss = F.mse_loss(input1, input2) + rou * loss_s
    return F.mse_loss(input1, input2), loss_s

def loss_smooth_energy(input1, input2, output, rou):
    _, _, h, w = output.size()
    dispx = output[0, 0, :, :]
    dispy = output[0, 0, :, :]
    loss_s = 0
    dispx_x = dispx.clone()
    dispy_x = dispy.clone()

    exx = dispx[0:h - 1, 0:w - 1] - dispx[0:h - 1, 1:w]
    exy = dispx[0:h - 1, 0:w - 1] - dispx[1:h, 0:w - 1]
    eyy = dispy[0:h - 1, 0:w - 1] - dispy[0:h - 1, 1:w]
    eyx = dispy[0:h - 1, 0:w - 1] - dispy[1:h, 0:w - 1]

    loss_s = (torch.sum(exx * exx) + torch.sum(eyy * eyy) + torch.sum(eyy * eyy) + 2 * torch.sum(exx * eyy) + torch.sum(
        exy * eyx)) / (w * h)
    # for x in range(0, w - 1):
    #     for y in range(0, h - 1):
    #         loss_s = loss_s + ((dispx[y, x] - dispx[y, x + 1]) ** 2 + (dispx[y, x] - dispx[y + 1, x]) ** 2 +
    #                            (dispy[y, x] - dispy[y, x + 1]) ** 2 + (dispy[y, x] - dispy[y + 1, x]) ** 2)
    # loss_s = (loss_s / (w * h)) ** 0.5
    # loss = F.mse_loss(input1, input2) + rou * loss_s
    return F.mse_loss(input1, input2), loss_s


def loss_smooth_4(input1, input2, output, rou):
    _, _, h, w = output.size()
    dispx = output[0, 0, :, :]
    dispy = output[0, 0, :, :]
    loss_s = 0
    dispx_x = dispx.clone()
    dispx_x2 = dispx.clone()
    dispy_x = dispy.clone()
    dispy_x2 = dispy.clone()
    dispx_x[:, 1:w] = dispx[:, 0:w - 1]
    dispy_x[:, 1:w] = dispy[:, 0:w - 1]
    dispx_x2[:, 0:w - 1] = dispx[:, 1:w]
    dispy_x2[:, 0:w - 1] = dispy[:, 1:w]
    dispx_y = dispx.clone()
    dispy_y = dispy.clone()
    dispx_y2 = dispx.clone()
    dispy_y2 = dispy.clone()
    dispx_y[1:w, :] = dispx[0:w - 1, :]
    dispy_y[1:w, :] = dispy[0:w - 1, :]
    dispx_y[0:w - 1, :] = dispx[1:w, :]
    dispy_y[0:w - 1, :] = dispy[1:w, :]
    loss_s = F.mse_loss(dispx, dispx_x) + F.mse_loss(dispx, dispx_y) + \
             F.mse_loss(dispy, dispy_x) + F.mse_loss(dispy, dispy_y) + \
             F.mse_loss(dispx, dispx_x2) + F.mse_loss(dispx, dispx_y2) + \
             F.mse_loss(dispy, dispy_x2) + F.mse_loss(dispy, dispy_y2)
    # for x in range(0, w - 1):
    #     for y in range(0, h - 1):
    #         loss_s = loss_s + ((dispx[y, x] - dispx[y, x + 1]) ** 2 + (dispx[y, x] - dispx[y + 1, x]) ** 2 +
    #                            (dispy[y, x] - dispy[y, x + 1]) ** 2 + (dispy[y, x] - dispy[y + 1, x]) ** 2)
    # loss_s = (loss_s / (w * h)) ** 0.5
    # loss = F.mse_loss(input1, input2) + rou * loss_s
    return F.mse_loss(input1, input2), loss_s


def loss_smooth_2(input1, input2, output, rou):
    # 2-order
    _, _, h, w = output.size()
    dispx = output[0, 0, :, :]
    dispy = output[0, 0, :, :]
    loss_s = 0
    dispx_x0 = dispx.clone()
    dispx_x1 = dispx.clone()
    dispy_x0 = dispy.clone()
    dispy_x1 = dispy.clone()
    dispx_y0 = dispx.clone()
    dispx_y1 = dispx.clone()
    dispy_y0 = dispy.clone()
    dispy_y1 = dispy.clone()
    dispx_x1[:, 1:w] = dispx[:, 0:w - 1]
    dispx_x0[:, 0:w - 1] = dispx[:, 1:w]
    dispy_x1[:, 1:w] = dispy[:, 0:w - 1]
    dispy_x0[:, 0:w - 1] = dispy[:, 1:w]

    dispx_y1[:, 1:w] = dispx[:, 0:w - 1]
    dispx_y0[1:w, :] = dispx[0:w - 1, :]
    dispy_y1[:, 1:w] = dispy[:, 0:w - 1]
    dispy_y0[1:w, :] = dispy[0:w - 1, :]

    dexx_2 = 2 * dispx - (dispx_x0 + dispx_x1)
    dexy_2 = 2 * dispx - (dispx_y0 + dispx_y1)
    deyx_2 = 2 * dispy - (dispy_x0 + dispy_x1)
    deyy_2 = 2 * dispy - (dispy_y0 + dispy_y1)
    loss_s = torch.norm(dexx_2, 2) + torch.norm(dexy_2, 2) + \
             torch.norm(deyx_2, 2) + torch.norm(deyy_2, 2)
    # for x in range(0, w - 1):
    #     for y in range(0, h - 1):
    #         loss_s = loss_s + ((dispx[y, x] - dispx[y, x + 1]) ** 2 + (dispx[y, x] - dispx[y + 1, x]) ** 2 +
    #                            (dispy[y, x] - dispy[y, x + 1]) ** 2 + (dispy[y, x] - dispy[y + 1, x]) ** 2)
    # loss_s = (loss_s / (w * h)) ** 0.5
    # loss = F.mse_loss(input1, input2) + rou * loss_s
    return F.mse_loss(input1, input2), loss_s / (w * h)


def loss_smooth_1and2(input1, input2, output, rou):
    # 2-order
    _, _, h, w = output.size()
    dispx = output[0, 0, :, :]
    dispy = output[0, 0, :, :]
    loss_s = 0
    dispx_x0 = dispx.clone()
    dispx_x1 = dispx.clone()
    dispy_x0 = dispy.clone()
    dispy_x1 = dispy.clone()
    dispx_y0 = dispx.clone()
    dispx_y1 = dispx.clone()
    dispy_y0 = dispy.clone()
    dispy_y1 = dispy.clone()
    dispx_x1[:, 1:w] = dispx[:, 0:w - 1]
    dispx_x0[:, 0:w - 1] = dispx[:, 1:w]
    dispy_x1[:, 1:w] = dispy[:, 0:w - 1]
    dispy_x0[:, 0:w - 1] = dispy[:, 1:w]

    dispx_y1[:, 1:w] = dispx[:, 0:w - 1]
    dispx_y0[1:w, :] = dispx[0:w - 1, :]
    dispy_y1[:, 1:w] = dispy[:, 0:w - 1]
    dispy_y0[1:w, :] = dispy[0:w - 1, :]

    dexx_2 = 2 * dispx - (dispx_x0 + dispx_x1)
    dexy_2 = 2 * dispx - (dispx_y0 + dispx_y1)
    deyx_2 = 2 * dispy - (dispy_x0 + dispy_x1)
    deyy_2 = 2 * dispy - (dispy_y0 + dispy_y1)
    loss_s2 = (torch.norm(dexx_2, 2) + torch.norm(dexy_2, 2) + \
               torch.norm(deyx_2, 2) + torch.norm(deyy_2, 2)) / (w * h)
    loss_s1 = F.mse_loss(dispx, dispx_x1) + F.mse_loss(dispx, dispx_y1) + \
              F.mse_loss(dispy, dispy_x1) + F.mse_loss(dispy, dispy_y1)
    # for x in range(0, w - 1):
    #     for y in range(0, h - 1):
    #         loss_s = loss_s + ((dispx[y, x] - dispx[y, x + 1]) ** 2 + (dispx[y, x] - dispx[y + 1, x]) ** 2 +
    #                            (dispy[y, x] - dispy[y, x + 1]) ** 2 + (dispy[y, x] - dispy[y + 1, x]) ** 2)
    # loss_s = (loss_s / (w * h)) ** 0.5
    # loss = F.mse_loss(input1, input2) + rou * loss_s
    return F.mse_loss(input1, input2), loss_s1, loss_s2


def loss_smooth_patch(input1, input2, output, rou):
    # 2-order
    _, _, h, w = output.size()
    dispx = output[0, 0, :, :]
    dispy = output[0, 0, :, :]
    loss_s = 0
    dispx_x0 = dispx.clone()
    dispx_x1 = dispx.clone()
    dispy_x0 = dispy.clone()
    dispy_x1 = dispy.clone()
    dispx_y0 = dispx.clone()
    dispx_y1 = dispx.clone()
    dispy_y0 = dispy.clone()
    dispy_y1 = dispy.clone()
    dispx_x1[:, 1:w] = dispx[:, 0:w - 1]
    dispx_x0[:, 0:w - 1] = dispx[:, 1:w]
    dispy_x1[:, 1:w] = dispy[:, 0:w - 1]
    dispy_x0[:, 0:w - 1] = dispy[:, 1:w]

    dispx_y1[:, 1:w] = dispx[:, 0:w - 1]
    dispx_y0[1:w, :] = dispx[0:w - 1, :]
    dispy_y1[:, 1:w] = dispy[:, 0:w - 1]
    dispy_y0[1:w, :] = dispy[0:w - 1, :]

    dexx_2 = 2 * dispx - (dispx_x0 + dispx_x1)
    dexy_2 = 2 * dispx - (dispx_y0 + dispx_y1)
    deyx_2 = 2 * dispy - (dispy_x0 + dispy_x1)
    deyy_2 = 2 * dispy - (dispy_y0 + dispy_y1)
    loss_s2 = (torch.norm(dexx_2, 2) + torch.norm(dexy_2, 2) + \
               torch.norm(deyx_2, 2) + torch.norm(deyy_2, 2)) / (w * h)
    loss_s1 = F.mse_loss(dispx, dispx_x1) + F.mse_loss(dispx, dispx_y1) + \
              F.mse_loss(dispy, dispy_x1) + F.mse_loss(dispy, dispy_y1)
    # for x in range(0, w - 1):
    #     for y in range(0, h - 1):
    #         loss_s = loss_s + ((dispx[y, x] - dispx[y, x + 1]) ** 2 + (dispx[y, x] - dispx[y + 1, x]) ** 2 +
    #                            (dispy[y, x] - dispy[y, x + 1]) ** 2 + (dispy[y, x] - dispy[y + 1, x]) ** 2)
    # loss_s = (loss_s / (w * h)) ** 0.5
    # loss = F.mse_loss(input1, input2) + rou * loss_s
    return F.mse_loss(input1, input2), loss_s1, loss_s2
