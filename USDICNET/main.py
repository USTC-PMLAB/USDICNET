import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import pandas as pd
from multiscaleloss import loss_smooth
from multiscaleloss import *
import models
import datasets
import numpy as np
import shutil
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import glob
from imageio import imread
import matplotlib.pyplot as plt

best_loss = -1

dataset_names = sorted(name for name in datasets.__all__)

n_iter = 0
w = 256
h = 256
# device = torch.device("cpu")
# os.environ[‘CUDA_VISIBLE_DEVICES’] = ‘0,1’
world_size = 1  # torch.cuda.device_count()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='U-DICNet Training on speckle dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--arch', default='U_DICNet', choices=['StrainNet_f', 'U_DICNet', 'U_StrainNet_f'],
                    help='network selection')
# parser.add_argument('--train_dataset_root', '-trr', metavar='DIR',
#                     help='path to training dataset')
# parser.add_argument('--test_dataset_root', '-ter', metavar='DIR',
#                     help='path to training dataset')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--solver', default='sgd', choices=['adam', 'sgd'],
                    help='solver algorithms')
parser.add_argument('--train', default='train_patch_shape1',
                    choices=['train', 'train_patch_shape1'],
                    help='train function')
parser.add_argument('--loss', default='loss_smooth',
                    choices=['norm2', 'abs_mean', 'norm1', 'MSE', 'loss_smooth', 'loss_smooth_1and2', 'loss_smooth_en'],
                    help='loss function')
parser.add_argument('--rou', default=0.1, type=float,
                    help='configure of loss_smooth in loss function')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--milestones', default=[40, 80, 120, 160, 200, 240],
                    nargs='*', help='epochs at which learning rate is divided by 2')
# parser.add_argument('--milestones', default=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280],
#                     metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')

parser.add_argument('--weights', '-w', default=[0.24, 0.08, 0.05, 0.02, 0.24], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))  # 0.01, 0.02, 0.05, 0.08, 0.24

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rank = 0


# rou = args.rou


# device = torch.device("cuda:1")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def main():
    global args, best_loss, w, h, device
    args = parser.parse_args()
    # Pictures' root 图片所在目录
    imgs_path: str = '/home/zhcy/PycharmProjects/DICNet_P/images/e_r1d0.6/'

    save_path = imgs_path
    # Read the picture and generate the input tensor 读取图片并生成输入张量
    refimg_list = sorted(glob.glob(os.path.join(imgs_path, 're*.bmp')))
    tarimg_list = sorted(glob.glob(os.path.join(imgs_path, 'tar*.bmp')))

    refimg_name = refimg_list[0]
    tarimg_name = tarimg_list[0]

    re_img = imread(refimg_name)
    tar_img = imread(tarimg_name)


    u_list = sorted(glob.glob(os.path.join(imgs_path, 'u*.csv')))
    v_list = sorted(glob.glob(os.path.join(imgs_path, 'v*.csv')))
    u_name = u_list[0]
    v_name = v_list[0]
    u = np.array(pd.read_csv(u_name, header=None))
    v = np.array(pd.read_csv(v_name, header=None))


    # Picture normalization 图片归一化
    # re_img = (re_img - np.mean(re_img)) / np.max(np.abs(re_img - np.mean(re_img)))
    # tar_img = (tar_img - np.mean(tar_img)) / np.max(np.abs(tar_img - np.mean(tar_img)))
    re_img = re_img / 255 * 10
    tar_img = tar_img / 255 * 10
    input_ref = torch.from_numpy(re_img).float()
    input_tar = torch.from_numpy(tar_img).float()
    input = torch.stack((input_ref, input_tar), 0).unsqueeze(0)
    w = input.shape[3]
    h = input.shape[2]
    print(w, h)
    # 是否加载预训练模型
    if args.pretrained:
        # using pre-trained model
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model ")
        best_loss = network_data['best_loss']
    else:
        network_data = None
        print("=> creating new model ")
    # creat model 定义模型

    #loss4-2
    # Whether to load pre-training 加载预训练
    # pretrained_path = '/home/zhcy/PycharmProjects/DICNet_P/data_U_DICNet_network_data/300epochs,b32,lr1e-05/model_best.pth.tar'
    # network_data = torch.load(pretrained_path)

    # model = models.__dict__['U_DICNet'](network_data).to(device)
    model = models.__dict__['U_DICNet'](network_data).to(device)  # , drop=False)

    #loss4-1
    # model = models.__dict__['U_DICNet_shape2'](network_data).to(device)  # , drop=False)


    # training parameters
    param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-9)
    writer = SummaryWriter(os.path.join(save_path, 'train'))
    # start 开始
    start = time.time()
    tar_img_1 = input_tar.unsqueeze(0).unsqueeze(0)
    plt.figure(101)
    for epoch in range(args.start_epoch, args.epochs):


        # loss = train_patch_shape2(input, model, optimizer, scheduler, device) # loss4-1
        loss = train_patch_shape2_ori(input, model, optimizer, scheduler, device) #loss4-2


        writer.add_scalar('loss', loss, epoch)

        if epoch % 10 == 0:
            print(epoch)
            print(loss)


    time_used = (time.time() - start)
    print(time_used)
    filename = 'checkpoint.pth.tar'
    torch.save({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_EPE': best_loss,
    }, os.path.join(save_path, filename))

    model.eval()
    input = input.to(device)
    output = model(input)
    output_to_write = output.data.cpu()
    output_to_write = output_to_write.numpy()
    disp = output_to_write
    disp_x = disp[0, 0, :, :]
    if output.size(1) == 12:
        disp_y = disp[0, 6, :, :]
    else:
        disp_y = disp[0, 1, :, :]
    # save the result
    # np.savetxt(save_path + '/dispx_' + img2_file[-7:-4] + sufixx + '.csv', disp_x[:, :], delimiter=',')
    # np.savetxt(save_path + '/dispy_' + img2_file[-7:-4] + sufixx + '.csv', disp_y[:, :], delimiter=',')
    np.savetxt(save_path + '/dispx_' + refimg_name[-9:-4] + '.csv', disp_x[:, :], delimiter=',')
    np.savetxt(save_path + '/dispy_' + refimg_name[-9:-4] + '.csv', disp_y[:, :], delimiter=',')

    input_ref = input[0, 0, :, :].unsqueeze(0).unsqueeze(0)
    input_tar = input[0, 1, :, :].unsqueeze(0).unsqueeze(0)
    disp_final0 = model(input)
    disp_final = disp_final0[:, 0:2, :, :]
    # disp_x = disp[0, 0, :, :]
    # disp_y = disp[0, 1, :, :]
    # 求loss
    output_ref = interpolation(input_tar, disp_final, device)
    tar_img2 = output_ref[0, 0, :, :].cpu()
    tar_img2 = tar_img2.detach().numpy()

    # draw pictures

    umin = np.min(u)
    umax = np.max(u)
    vmin = np.min(v)
    vmax = np.max(v)

    plt.figure(1)
    re1 = plt.subplot(4, 2, 1)
    plt.imshow(re_img, cmap='gray')
    tar1 = plt.subplot(4, 2, 2)
    plt.imshow(tar_img, cmap='gray')

    u2 = plt.subplot(4, 2, 3)
    plt.imshow(disp_x, cmap='jet', vmin=umin, vmax=umax)
    plt.colorbar()
    v2 = plt.subplot(4, 2, 4)
    plt.imshow(disp_y, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    # plt.show()


    plt.show()


#loss4-1
def train_patch_shape2(input, model, optimizer, scheduler, device):
    global args, w, h
    model.train()
    input = input.to(device)
    input_ref = input[0, 0, :, :].unsqueeze(0).unsqueeze(0)
    input_tar = input[0, 1, :, :].unsqueeze(0).unsqueeze(0)
    disp = model(input)
    dispuxy = disp[0, 0:6, :, :]
    dispvxy = disp[0, 6:12, :, :]

    # loss
    loss = torch.tensor(0).to(device)
    r = 2
    order = 2

    # （u=(XX.*COEFFS)*difs(u,x,y),v=(XX.*COEFFS)*difs(v,x,y)）
    # xx = torch.tensor(np.zeros(r + 1, r + 1))
    # torch.matmul()
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            # loss1 = torch.tensor(0).to(device)
            # xx[i, j,:] = [i ** (n - m) * j ** m for n in range(order + 1) for m in range(n + 1)]
            xx = torch.tensor([i ** (n - m) * j ** m for n in range(order + 1) for m in range(n + 1)]).to(device)
            Coffes = torch.tensor([1, 1, 1, 1 / 2, 1 / 2, 1 / 2]).to(device)
            # xxx = np.multiply(xx, Coffes)
            xxx = torch.mul(xx, Coffes).to(device)
            # disp[i, j] = torch.matmul(xxx, )
            dispu = torch.einsum('i,ijk->jk', xxx, dispuxy)
            dispv = torch.einsum('i,ijk->jk', xxx, dispvxy)
            dispuv = torch.stack((dispu, dispv), dim=0).unsqueeze(0)

            output = interpolation(input_tar[:, :, r + j:h - r + j, r + i:w - r + i], dispuv[:, :, r:h - r, r:w - r],
                                   device)
            loss = loss + F.mse_loss(output, input_ref[:, :, r + j:h - r + j, r + i:w - r + i]) / ((r +1) ** 2)
            # loss = loss + F.mse_loss(output, input_ref[:, :, r + n:h - r + n, r + m:w - r + m]) / ((r + 1) ** 2)/2/(i**2+j**2)
            # loss1 = F.mse_loss(output, input_ref[:, :, r + j:h - r + j, r + i:w - r + i]) / ((r + 1) ** 2)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # scheduler.step()
    scheduler.step(loss)
    return loss

# loss4-2
def train_patch_shape2_ori(input, model, optimizer, scheduler, device):
    global args, w, h
    model.train()
    input = input.to(device)
    input_ref = input[0, 0, :, :].unsqueeze(0).unsqueeze(0)
    input_tar = input[0, 1, :, :].unsqueeze(0).unsqueeze(0)
    disp = model(input)

    dispu = disp[:, 0, :, :].unsqueeze(1)
    dispv = disp[:, 1, :, :].unsqueeze(1)

    dispux = torch.gradient(dispu, dim=3)[0]
    dispuy = torch.gradient(dispu, dim=2)[0]
    dispuxx = torch.gradient(dispux, dim=3)[0]
    dispuxy = torch.gradient(dispux, dim=2)[0]
    # dispuyx = torch.gradient(dispux, dim=3)
    dispuyy = torch.gradient(dispuy, dim=2)[0]

    dispvx = torch.gradient(dispv, dim=3)[0]
    dispvy = torch.gradient(dispv, dim=2)[0]
    dispvxx = torch.gradient(dispvx, dim=3)[0]
    dispvxy = torch.gradient(dispvx, dim=2)[0]
    # dispvyx = torch.gradient(dispvx, dim=3)
    dispvyy = torch.gradient(dispvy, dim=2)[0]

    dispu_xy = torch.cat([dispu, dispux, dispuy, dispuxx, dispuxy, dispuyy], dim=1).squeeze(0)
    dispv_xy = torch.cat([dispv, dispvx, dispvy, dispvxx, dispvxy, dispvyy], dim=1).squeeze(0)
    # loss
    loss = torch.tensor(0).to(device)
    r = 2
    order = 2

    # （u=(XX.*COEFFS)*difs(u,x,y),v=(XX.*COEFFS)*difs(v,x,y)）
    # xx = torch.tensor(np.zeros(r + 1, r + 1))
    # torch.matmul()
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            # loss1 = torch.tensor(0).to(device)
            # xx[i, j,:] = [i ** (n - m) * j ** m for n in range(order + 1) for m in range(n + 1)]
            xx = torch.tensor([i ** (n - m) * j ** m for n in range(order + 1) for m in range(n + 1)]).to(device)
            Coffes = torch.tensor([1, 1, 1, 1 / 2, 1 / 2, 1 / 2]).to(device)
            # xxx = np.multiply(xx, Coffes)
            xxx = torch.mul(xx, Coffes).to(device)
            # disp[i, j] = torch.matmul(xxx, )
            dispu = torch.einsum('i,ijk->jk', xxx, dispu_xy)
            dispv = torch.einsum('i,ijk->jk', xxx, dispv_xy)
            dispuv = torch.stack((dispu, dispv), dim=0).unsqueeze(0)

            output = interpolation(input_tar[:, :, r + j:h - r + j, r + i:w - r + i], dispuv[:, :, r:h - r, r:w - r],
                                   device)
            loss = loss + F.mse_loss(output, input_ref[:, :, r + j:h - r + j, r + i:w - r + i]) / ((r +1 ) ** 2)/2


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # scheduler.step()
    scheduler.step(loss)
    return loss



def one_scale(disp, input_ref, input_tar, device):
    global w, h

    disp2 = F.interpolate(disp, size=[128, 128], mode='bicubic')
    output_ref = interpolation(input_tar, disp2, device)
    # loss_map = output_ref - input_ref
    loss = F.mse_loss(output_ref, input_ref)
    return loss


def interpolation(feature_map, flow, device):
    b, c, w, h = feature_map.size()
    a_x = torch.linspace(-1, 1, w, device=device)
    a_y = torch.linspace(-1, 1, h, device=device)
    # b = a.copy
    y, x = torch.meshgrid(a_x, a_y)
    x = x.repeat(b, 1, 1)
    y = y.repeat(b, 1, 1)

    deform_x = (x + flow[:, 0, :, :] / w * 2)

    deform_y = (y + flow[:, 1, :, :] / h * 2)

    coordiante = torch.stack((deform_x, deform_y), 3)

    interp_featuremap = F.grid_sample(feature_map, coordiante, mode='bicubic', padding_mode='border',
                                      align_corners=True)
    return interp_featuremap


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
