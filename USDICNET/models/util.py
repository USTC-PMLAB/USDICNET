"""Basic layer builders for U-DICNet.

These helpers are kept numerically identical to the original implementation;
only formatting and docstrings have been cleaned up.
"""
import torch.nn as nn


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, drop=False):
    """Conv -> [BN] -> [Dropout] -> LeakyReLU(0.1) block.

    Args:
        batchNorm: apply BatchNorm2d when True, otherwise use bias in Conv2d.
        in_planes: input channel count.
        out_planes: output channel count.
        kernel_size: conv kernel size, padding is `(kernel_size - 1) // 2` to keep size.
        stride: conv stride.
        drop: add Dropout2d(0.4) when True.
    """
    padding = (kernel_size - 1) // 2
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, padding=padding, bias=not batchNorm)]
    if batchNorm:
        layers.append(nn.BatchNorm2d(out_planes))
    if drop:
        layers.append(nn.Dropout2d(0.4))
    layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*layers)


def predict_flow(in_planes, out_planes, drop=False):
    """1x1-ish prediction conv (kernel 3, stride 1, pad 1, no bias)."""
    if drop:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout2d(0.4),
        )
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def deconv(in_planes, out_planes, stride=2, drop=False):
    """Transposed conv (kernel 4, pad 1) -> [Dropout] -> LeakyReLU(0.1)."""
    layers = [nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                                  stride=stride, padding=1, bias=False)]
    if drop:
        layers.append(nn.Dropout2d(0.4))
    layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*layers)


def crop_like(input_img, target):
    """Crop ``input_img`` spatial dims to match ``target`` when they differ."""
    if input_img.size()[2:] == target.size()[2:]:
        return input_img
    return input_img[:, :, :target.size(2), :target.size(3)]
