#!/usr/bin/env python3
"""U‑DICNet — unsupervised DIC displacement measurement from a single image pair.

This is a cleaned‑up version of the original main.py.  All training logic,
loss functions, and output formats are kept numerically identical.

Usage::

    # default (U_DICNet, patch_grad loss)
    python main.py --data-dir ./gauss_displacement

    # 12‑channel network with patch12 loss
    python main.py --data-dir ./star_displacement --arch U_DICNet_shape2 --loss patch12
"""
import argparse
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from imageio import imread
from torch.utils.tensorboard import SummaryWriter

import models  # noqa: F401  — registers U_DICNet / U_DICNet_shape2

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class AverageMeter:
    """Computes and stores the average and current value."""

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
        return f'{self.val:.3f} ({self.avg:.3f})'


def interpolation(feature_map, flow, device):
    """Warp ``feature_map`` by ``flow`` via bilinear grid_sample."""
    b, c, h, w = feature_map.size()
    a_y = torch.linspace(-1, 1, h, device=device)
    a_x = torch.linspace(-1, 1, w, device=device)
    y, x = torch.meshgrid(a_y, a_x, indexing='ij')
    x = x.repeat(b, 1, 1)
    y = y.repeat(b, 1, 1)

    deform_x = x + flow[:, 0, :, :] / w * 2
    deform_y = y + flow[:, 1, :, :] / h * 2
    coordinate = torch.stack((deform_x, deform_y), 3)
    return F.grid_sample(feature_map, coordinate, mode='bicubic',
                         padding_mode='border', align_corners=True)


# ---------------------------------------------------------------------------
# loss: patch12 — 12‑channel output, second‑order Taylor expansion (loss4‑1)
# ---------------------------------------------------------------------------
def train_patch_shape2(input_ref, input_tar, disp_6ch, radius, order, device):
    """Loss from 12‑ch output (u, v, u_x, v_x, u_y, v_y, u_xx, v_xx, u_xy, v_xy, u_yy, v_yy).

    Coefs: [1, 1, 1, 1/2, 1/2, 1/2] — kept identical to original.
    """
    h, w = input_ref.shape[2], input_ref.shape[3]
    dispuxy = disp_6ch[0, 0:6, :, :]
    dispvxy = disp_6ch[0, 6:12, :, :]

    coeffs = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]).to(device)
    loss = torch.tensor(0.0).to(device)

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            xx = torch.tensor(
                [i ** (n - m) * j ** m for n in range(order + 1) for m in range(n + 1)]
            ).to(device)
            xxx = torch.mul(xx, coeffs)

            dispu = torch.einsum('i,ijk->jk', xxx, dispuxy)
            dispv = torch.einsum('i,ijk->jk', xxx, dispvxy)
            dispuv = torch.stack((dispu, dispv), dim=0).unsqueeze(0)

            output = interpolation(
                input_tar[:, :, radius + j:h - radius + j, radius + i:w - radius + i],
                dispuv[:, :, radius:h - radius, radius:w - radius],
                device,
            )
            loss = loss + F.mse_loss(
                output,
                input_ref[:, :, radius + j:h - radius + j, radius + i:w - radius + i],
            ) / ((radius + 1) ** 2)

    return loss


# ---------------------------------------------------------------------------
# loss: patch_grad — 2‑ch output + numerical gradients (loss4‑2)
# ---------------------------------------------------------------------------
def train_patch_shape2_ori(input_ref, input_tar, disp, radius, order, device):
    """Compute numerical 1st/2nd‑order gradients of (u,v), then apply the same
    second‑order Taylor‑expansion loss as ``train_patch_shape2``."""
    h, w = input_ref.shape[2], input_ref.shape[3]

    dispu = disp[:, 0, :, :].unsqueeze(1)
    dispv = disp[:, 1, :, :].unsqueeze(1)

    dispux  = torch.gradient(dispu, dim=3)[0]
    dispuy  = torch.gradient(dispu, dim=2)[0]
    dispuxx = torch.gradient(dispux, dim=3)[0]
    dispuxy = torch.gradient(dispux, dim=2)[0]
    dispuyy = torch.gradient(dispuy, dim=2)[0]

    dispvx  = torch.gradient(dispv, dim=3)[0]
    dispvy  = torch.gradient(dispv, dim=2)[0]
    dispvxx = torch.gradient(dispvx, dim=3)[0]
    dispvxy = torch.gradient(dispvx, dim=2)[0]
    dispvyy = torch.gradient(dispvy, dim=2)[0]

    dispu_xy = torch.cat([dispu, dispux, dispuy, dispuxx, dispuxy, dispuyy], dim=1).squeeze(0)
    dispv_xy = torch.cat([dispv, dispvx, dispvy, dispvxx, dispvxy, dispvyy], dim=1).squeeze(0)

    coeffs = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]).to(device)
    loss = torch.tensor(0.0).to(device)

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            xx = torch.tensor(
                [i ** (n - m) * j ** m for n in range(order + 1) for m in range(n + 1)]
            ).to(device)
            xxx = torch.mul(xx, coeffs)

            dispu_ij = torch.einsum('i,ijk->jk', xxx, dispu_xy)
            dispv_ij = torch.einsum('i,ijk->jk', xxx, dispv_xy)
            dispuv = torch.stack((dispu_ij, dispv_ij), dim=0).unsqueeze(0)

            output = interpolation(
                input_tar[:, :, radius + j:h - radius + j, radius + i:w - radius + i],
                dispuv[:, :, radius:h - radius, radius:w - radius],
                device,
            )
            loss = loss + F.mse_loss(
                output,
                input_ref[:, :, radius + j:h - radius + j, radius + i:w - radius + i],
            ) / ((radius + 1) ** 2) / 2

    return loss


# ---------------------------------------------------------------------------
# training step dispatcher
# ---------------------------------------------------------------------------
def train_step(loss_type, input_ref, input_tar, disp, radius, order, device):
    """Run one forward+backward pass. Returns scalar loss (float)."""
    if loss_type == 'patch12':
        loss = train_patch_shape2(input_ref, input_tar, disp, radius, order, device)
    elif loss_type == 'patch_grad':
        loss = train_patch_shape2_ori(input_ref, input_tar, disp, radius, order, device)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss


# ---------------------------------------------------------------------------
# image loading
# ---------------------------------------------------------------------------
def load_image_pair(data_dir, norm_factor=10.0):
    """Load the first reference/target .bmp pair, normalise to [0, norm_factor]."""
    ref_list = sorted(glob.glob(os.path.join(data_dir, 're*.bmp')))
    tar_list = sorted(glob.glob(os.path.join(data_dir, 'tar*.bmp')))
    if not ref_list or not tar_list:
        raise FileNotFoundError(f"No re*.bmp / tar*.bmp found in {data_dir}")

    re_img = imread(ref_list[0]) / 255.0 * norm_factor
    tar_img = imread(tar_list[0]) / 255.0 * norm_factor

    ref_t = torch.from_numpy(re_img).float()
    tar_t = torch.from_numpy(tar_img).float()
    input_tensor = torch.stack((ref_t, tar_t), 0).unsqueeze(0)  # (1,2,H,W)
    return ref_list[0], tar_list[0], re_img, tar_img, input_tensor


# ---------------------------------------------------------------------------
# model / optimizer builders
# ---------------------------------------------------------------------------
def build_model(arch, pretrained, device):
    """Instantiate the network, optionally loading a checkpoint."""
    if pretrained:
        network_data = torch.load(pretrained, map_location=device)
        print(f"=> using pre-trained model: {pretrained}")
    else:
        network_data = None
        print("=> creating new model")

    model = models.__dict__[arch](network_data).to(device)
    return model


def build_optimizer(model, lr, args):
    """Build optimiser + scheduler (ReduceLROnPlateau by default)."""
    param_groups = [
        {'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
        {'params': model.weight_parameters(), 'weight_decay': args.weight_decay},
    ]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr,
                                    momentum=args.momentum)
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-9,
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# main training loop (with adaptive lr retry)
# ---------------------------------------------------------------------------
def train_loop(input_tensor, model, args, device, optimizer, scheduler):
    """Core training loop (ReduceLROnPlateau, identical to original algorithm)."""
    input_tensor = input_tensor.to(device)
    input_ref = input_tensor[0, 0, :, :].unsqueeze(0).unsqueeze(0)
    input_tar = input_tensor[0, 1, :, :].unsqueeze(0).unsqueeze(0)

    loss_val = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        disp = model(input_tensor)

        loss = train_step(args.loss, input_ref, input_tar, disp,
                          args.radius, args.order, device)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss)

        loss_val = loss.item()
        args.writer.add_scalar('loss', loss_val, epoch)

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"epoch {epoch:5d}  lr {current_lr:.2e}  loss {loss_val:.6f}")

    return loss_val


# ---------------------------------------------------------------------------
# adaptive lr retry variant (optional, enabled via --auto-retry)
# ---------------------------------------------------------------------------
def train_loop_retry(input_tensor, model_factory, args, device):
    """Training loop that auto-detects stalled convergence and re-initialises
    with a doubled learning rate (up to max_retries times)."""
    check_interval = 100
    threshold = 0.80       # first 100 epochs must drop >= 80% from initial loss
    max_retries = 3

    input_tensor = input_tensor.to(device)
    input_ref = input_tensor[0, 0, :, :].unsqueeze(0).unsqueeze(0)
    input_tar = input_tensor[0, 1, :, :].unsqueeze(0).unsqueeze(0)

    current_lr = args.lr
    retry = 0
    optimizer = None
    scheduler = None

    while True:
        model = model_factory().to(device)
        optimizer, scheduler = build_optimizer(model, current_lr, args)
        initial_loss = None

        print(f"\n{'*'*60}")
        print(f"  retry {retry}/{max_retries}  lr = {current_lr:.2e}  model = {args.arch}")
        print(f"{'*'*60}")

        epoch_offset = retry * check_interval
        loss_val = 0.0

        for local_epoch in range(check_interval):
            global_epoch = epoch_offset + local_epoch
            model.train()
            disp = model(input_tensor)

            loss = train_step(args.loss, input_ref, input_tar, disp,
                              args.radius, args.order, device)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            loss_val = loss.item()
            if initial_loss is None:
                initial_loss = loss_val

            args.writer.add_scalar('loss', loss_val, global_epoch)

            if global_epoch % 10 == 0:
                current_lr_val = optimizer.param_groups[0]['lr']
                print(f"epoch {global_epoch:5d}  lr {current_lr_val:.2e}  loss {loss_val:.6f}")

        improvement = (initial_loss - loss_val) / initial_loss
        if improvement >= threshold:
            print(f"[retry {retry}] converged: loss {initial_loss:.4f} -> {loss_val:.4f}  "
                  f"(delta = {improvement*100:.1f}% >= {threshold*100:.0f}%)  OK  continue training\n")
            break
        else:
            print(f"[retry {retry}] stalled:   loss {initial_loss:.4f} -> {loss_val:.4f}  "
                  f"(delta = {improvement*100:.1f}% < {threshold*100:.0f}%)  FAIL")
            retry += 1
            if retry > max_retries:
                print(f"[ERROR] max retries ({max_retries}) reached.  Training may not converge.")
                break
            current_lr *= 2.0

    remaining = args.epochs - (retry * check_interval + check_interval)
    if remaining > 0:
        print(f"continuing with lr={current_lr:.2e} for {remaining} remaining epochs...\n")
        for epoch in range(retry * check_interval + check_interval, args.epochs):
            model.train()
            disp = model(input_tensor)

            loss = train_step(args.loss, input_ref, input_tar, disp,
                              args.radius, args.order, device)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            loss_val = loss.item()
            args.writer.add_scalar('loss', loss_val, epoch)

            if epoch % 10 == 0:
                current_lr_val = optimizer.param_groups[0]['lr']
                print(f"epoch {epoch:5d}  lr {current_lr_val:.2e}  loss {loss_val:.6f}")

    return loss_val, model

def save_results(input_tensor, model, save_dir, ref_name, device, radius=2):
    """Save displacement CSVs and produce matplotlib figures.

    Edge ``radius`` pixels are excluded from colour‑bar limits and error
    reporting because the displacement field there is unreliable.
    """
    model.eval()

    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    output_np = output.data.cpu().numpy()

    disp_x_full = output_np[0, 0, :, :]
    if output.size(1) == 12:
        disp_y_full = output_np[0, 6, :, :]
    else:
        disp_y_full = output_np[0, 1, :, :]

    # ROI: exclude edge pixels where the loss was not evaluated
    h, w = disp_x_full.shape
    r = radius
    disp_x_roi = disp_x_full[r:h - r, r:w - r]
    disp_y_roi = disp_y_full[r:h - r, r:w - r]

    # CSV output — full field (same as original)
    basename = os.path.basename(ref_name)
    tag = basename.replace('re', '').replace('.bmp', '')
    np.savetxt(os.path.join(save_dir, f'dispx_{tag}.csv'), disp_x_full, delimiter=',')
    np.savetxt(os.path.join(save_dir, f'dispy_{tag}.csv'), disp_y_full, delimiter=',')

    # --- visualisation ---
    ref_np = input_tensor[0, 0, :, :].cpu().numpy()
    tar_np = input_tensor[0, 1, :, :].cpu().numpy()

    # colour‑bar limits from valid region only
    umin, umax = float(np.min(disp_x_roi)), float(np.max(disp_x_roi))
    vmin, vmax = float(np.min(disp_y_roi)), float(np.max(disp_y_roi))

    plt.figure(1, figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(ref_np, cmap='gray')
    plt.title('Reference')

    plt.subplot(2, 2, 2)
    plt.imshow(tar_np, cmap='gray')
    plt.title('Target')

    plt.subplot(2, 2, 3)
    plt.imshow(disp_x_full, cmap='jet', vmin=umin, vmax=umax)
    plt.colorbar()
    plt.title(f'disp_x  (ROI: [{umin:.3f}, {umax:.3f}])')

    plt.subplot(2, 2, 4)
    plt.imshow(disp_y_full, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f'disp_y  (ROI: [{vmin:.3f}, {vmax:.3f}])')

    png_path = os.path.join(save_dir, 'result_figure.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"result figure saved to {png_path}")

    # ROI stats for console
    print(f"  disp_x ROI: [{umin:.4f}, {umax:.4f}]")
    print(f"  disp_y ROI: [{vmin:.4f}, {vmax:.4f}]")
    return disp_x_full, disp_y_full


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='U‑DICNet — unsupervised DIC with single‑pair image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- paths ---
    parser.add_argument('--data-dir', default='./gauss_displacement',
                        help='Directory containing re*.bmp / tar*.bmp')

    # --- model ---
    parser.add_argument('--arch', default='U_DICNet',
                        choices=['U_DICNet', 'U_DICNet_shape2'])
    parser.add_argument('--pretrained', default=None,
                        help='path to pre-trained checkpoint')

    # --- loss ---
    parser.add_argument('--loss', default='patch_grad',
                        choices=['patch_grad', 'patch12'],
                        help='patch_grad = 2‑ch output + numerical gradients; '
                             'patch12 = 12‑ch output (shape2 network)')

    # --- solver ---
    parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=4e-4)
    parser.add_argument('--bias-decay', type=float, default=0.0)

    # --- training config ---
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--radius', type=int, default=2,
                        help='subset radius (pixels)')
    parser.add_argument('--order', type=int, default=2,
                        help='Taylor expansion order')
    parser.add_argument('--norm-factor', type=float, default=10.0,
                        help='image normalisation factor (pixel/255 * factor)')
    parser.add_argument('--save-dir', default=None,
                        help='output directory (default: same as data-dir)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')
    parser.add_argument('--auto-retry', action='store_true', default=False,
                        help='enable adaptive lr retry on stalled convergence')

    args = parser.parse_args()

    # seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # resolve save directory
    save_dir = args.save_dir or args.data_dir
    os.makedirs(save_dir, exist_ok=True)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load images
    ref_name, tar_name, re_img, tar_img, input_tensor = load_image_pair(
        args.data_dir, args.norm_factor,
    )
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    print(f"image size: {w} × {h}")
    if args.seed is not None:
        print(f"random seed: {args.seed}")
    else:
        print("random seed: not set (results may vary; use --seed for reproducibility)")

    # writer
    writer = SummaryWriter(os.path.join(save_dir, 'train'))
    args.writer = writer

    # model
    model = build_model(args.arch, args.pretrained, device)

    # train
    start = time.time()
    print(f"training {args.arch} with loss={args.loss}, radius={args.radius}, "
          f"order={args.order}, epochs={args.epochs}")
    if args.auto_retry:
        print("[auto-retry] enabled -- will re-init if loss stalls")
        model_factory = lambda: build_model(args.arch, args.pretrained, device)
        final_loss, model = train_loop_retry(input_tensor, model_factory, args, device)
    else:
        optimizer, scheduler = build_optimizer(model, args.lr, args)
        final_loss = train_loop(input_tensor, model, args, device, optimizer, scheduler)
    elapsed = time.time() - start
    print(f"training finished in {elapsed:.1f}s, final loss = {final_loss:.6f}")

    # checkpoint
    ckpt_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save({
        'epoch': args.epochs,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_EPE': -1,
    }, ckpt_path)
    print(f"checkpoint saved to {ckpt_path}")

    writer.close()

    # save results & plot
    disp_x, disp_y = save_results(input_tensor, model, save_dir, ref_name, device, args.radius)
    print(f"displacement CSVs saved to {save_dir}")


if __name__ == '__main__':
    main()
