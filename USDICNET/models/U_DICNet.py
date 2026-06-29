"""U-DICNet model variants for unsupervised DIC displacement measurement.

This file is a cleaned-up version of the original U_DICNet.py.  All layer
parameters, channel counts, initialisation, and forward-pass logic are kept
numerically identical to the original.
"""
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_

from .util import conv, predict_flow, deconv, crop_like

__all__ = ['StrainNetF', 'U_DICNet', 'U_DICNet_shape2']


# ---------------------------------------------------------------------------
# StrainNet‑F  (multi‑scale DIC backbone; input 6 ch = 3×|ref, tar|)
# ---------------------------------------------------------------------------
class StrainNetF(nn.Module):
    """StrainNet‑F encoder‑decoder with 5‑scale flow prediction.

    Args:
        batchNorm: apply BatchNorm2d after each convolution.
    """

    def __init__(self, batchNorm: bool = True):
        super().__init__()
        self.batchNorm = batchNorm

        # --- encoder ---
        self.conv1   = conv(batchNorm, 6,   64,   kernel_size=7, stride=1)
        self.conv2   = conv(batchNorm, 64,  128,  kernel_size=5, stride=1)
        self.conv3   = conv(batchNorm, 128, 256,  kernel_size=5, stride=2)
        self.conv3_1 = conv(batchNorm, 256, 256)
        self.conv4   = conv(batchNorm, 256, 512,  stride=2)
        self.conv4_1 = conv(batchNorm, 512, 512)
        self.conv5   = conv(batchNorm, 512, 512,  stride=2)
        self.conv5_1 = conv(batchNorm, 512, 512)
        self.conv6   = conv(batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(batchNorm, 1024, 1024)

        # --- decoder (transposed conv) ---
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770,  128)
        self.deconv2 = deconv(386,  64)

        # --- flow prediction heads ---
        self.predict_flow6 = predict_flow(1024, 2)
        self.predict_flow5 = predict_flow(1026, 2)
        self.predict_flow4 = predict_flow(770,  2)
        self.predict_flow3 = predict_flow(386,  2)
        self.predict_flow2 = predict_flow(194,  2)

        # --- upsampling for flow (×2 each level) ---
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # flow 6
        flow6    = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        # flow 5
        concat5   = torch.cat((out_conv5, out_deconv5, flow6_up), 1)    # 512+512+2=1026
        flow5     = self.predict_flow5(concat5)
        flow5_up  = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)        # 256

        # flow 4
        concat4   = torch.cat((out_conv4, out_deconv4, flow5_up), 1)    # 512+256+2=770
        flow4     = self.predict_flow4(concat4)
        flow4_up  = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)        # 128

        # flow 3
        concat3   = torch.cat((out_conv3, out_deconv3, flow4_up), 1)    # 256+128+2=386
        flow3     = self.predict_flow3(concat3)
        flow3_up  = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)        # 64

        # flow 2
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)      # 128+64+2=194
        flow2   = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        return flow2

    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if 'weight' in n]

    def bias_parameters(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]


# ---------------------------------------------------------------------------
# U‑StrainNet‑f  (same backbone, 2‑channel input = single image pair)
# ---------------------------------------------------------------------------
class U_StrainNet_f_model(nn.Module):
    """U‑StrainNet‑f: StrainNet‑F variant with 2‑channel input.

    Args:
        batchNorm: apply BatchNorm2d after each convolution.
    """

    def __init__(self, batchNorm: bool = True):
        super().__init__()
        self.batchNorm = batchNorm

        # --- encoder ---
        self.conv1   = conv(batchNorm, 2,   64,   kernel_size=7, stride=1)
        self.conv2   = conv(batchNorm, 64,  128,  kernel_size=5, stride=1)
        self.conv3   = conv(batchNorm, 128, 256,  kernel_size=5, stride=2)
        self.conv3_1 = conv(batchNorm, 256, 256)
        self.conv4   = conv(batchNorm, 256, 512,  stride=2)
        self.conv4_1 = conv(batchNorm, 512, 512)
        self.conv5   = conv(batchNorm, 512, 512,  stride=2)
        self.conv5_1 = conv(batchNorm, 512, 512)
        self.conv6   = conv(batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(batchNorm, 1024, 1024)

        # --- decoder ---
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770,  128)
        self.deconv2 = deconv(386,  64)

        # --- flow prediction heads ---
        self.predict_flow6 = predict_flow(1024, 2)
        self.predict_flow5 = predict_flow(1026, 2)
        self.predict_flow4 = predict_flow(770,  2)
        self.predict_flow3 = predict_flow(386,  2)
        self.predict_flow2 = predict_flow(194,  2)

        # --- upsampling ---
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # FIX: original had BatchNorm3d (typo), never triggered.
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # flow 6
        flow6    = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        # flow 5 — concat: 512+512+2=1026
        concat5   = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5     = self.predict_flow5(concat5)
        flow5_up  = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        # flow 4 — concat: 512+256+2=770
        concat4   = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4     = self.predict_flow4(concat4)
        flow4_up  = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        # flow 3 — concat: 256+128+2=386
        concat3   = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3     = self.predict_flow3(concat3)
        flow3_up  = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        # flow 2 — concat: 128+64+2=194
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2   = self.predict_flow2(concat2)

        if self.training:
            return flow2  # original returns single flow2 during training
        return flow2

    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if 'weight' in n]

    def bias_parameters(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]


# ---------------------------------------------------------------------------
# U‑DICNet  (default: 2‑ch output u,v)
# ---------------------------------------------------------------------------
class U_DICNet_model(nn.Module):
    """U‑DICNet: fine‑tuned U‑StrainNet‑f with 2‑channel displacement output.

    Overrides several layers of the *U_StrainNet_f_model* backbone to adapt
    to the DIC task (single-scale prediction, dropout on deeper layers).
    """

    def __init__(self, U_StrainNet_f_model, batchNorm: bool = True, drop: bool = False):
        super().__init__()
        self.model = U_StrainNet_f_model(batchNorm=batchNorm)
        self.batchNorm = batchNorm

        # --- layer overrides for fine‑tuning ---
        self.model.conv1 = conv(batchNorm, 2, 128, kernel_size=5, stride=1)
        self.model.conv2 = conv(batchNorm, 128, 128, kernel_size=3, stride=1)
        self.model.conv3 = conv(batchNorm, 128, 256, kernel_size=5, stride=1)
        self.model.conv5 = conv(batchNorm, 512, 512, stride=1)
        self.model.conv4_1 = conv(batchNorm, 512, 512, drop=drop)
        self.model.conv5_1 = conv(batchNorm, 512, 512, drop=drop)
        self.model.conv6_1 = conv(batchNorm, 1024, 1024, drop=drop)

        self.model.deconv4 = deconv(1026, 256, stride=1, drop=drop)
        self.model.deconv2 = deconv(386, 64, stride=1, drop=drop)

        self.model.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 1, 1, bias=False)
        self.model.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 1, 1, bias=False)

    def forward(self, x):
        return self.model(x)

    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if 'weight' in n]

    def bias_parameters(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]


# ---------------------------------------------------------------------------
# U‑DICNet shape2  (12‑ch output: u, u_x, u_y, u_xx, u_xy, u_yy × 2)
# ---------------------------------------------------------------------------
class U_DICNet_model_shape2(nn.Module):
    """U‑DICNet shape2: 12‑channel variant outputting displacement + 1st/2nd
    spatial derivatives for second‑order Taylor expansion in the loss."""

    def __init__(self, U_StrainNet_f_model, batchNorm: bool = True, drop: bool = False):
        super().__init__()
        self.model = U_StrainNet_f_model(batchNorm=batchNorm)
        self.batchNorm = batchNorm

        # --- layer overrides for fine‑tuning ---
        self.model.conv1 = conv(batchNorm, 2, 128, kernel_size=5, stride=1)
        self.model.conv2 = conv(batchNorm, 128, 128, kernel_size=3, stride=1)
        self.model.conv3 = conv(batchNorm, 128, 256, kernel_size=5, stride=1)
        self.model.conv5 = conv(batchNorm, 512, 512, stride=1)
        self.model.conv4_1 = conv(batchNorm, 512, 512, drop=drop)
        self.model.conv5_1 = conv(batchNorm, 512, 512, drop=drop)
        self.model.conv6_1 = conv(batchNorm, 1024, 1024, drop=drop)

        self.model.deconv4 = deconv(1026, 256, stride=1, drop=drop)
        self.model.deconv2 = deconv(386, 128, stride=1, drop=drop)

        # Final head: concat2 = conv2_out(128) + deconv2_out(128) + flow3_up(2) = 258 → 12
        self.model.predict_flow2 = predict_flow(258, 12)

        self.model.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 1, 1, bias=False)
        self.model.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 1, 1, bias=False)

    def forward(self, x):
        return self.model(x)

    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if 'weight' in n]

    def bias_parameters(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
def U_DICNet(data=None, batchNorm: bool = True, drop: bool = False):
    """Create a U‑DICNet (2‑channel output). Optionally load ``data['state_dict']``."""
    model = U_DICNet_model(U_StrainNet_f_model, batchNorm=batchNorm, drop=drop)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def U_DICNet_shape2(data=None, batchNorm: bool = True, drop: bool = False):
    """Create a U‑DICNet shape2 (12‑channel output). Optionally load ``data['state_dict']``."""
    model = U_DICNet_model_shape2(U_StrainNet_f_model, batchNorm=batchNorm, drop=drop)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model