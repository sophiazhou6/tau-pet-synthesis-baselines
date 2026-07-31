"""
3D Dense-U-Net — PyTorch port of the model used in
Neurology-AI-Program/AI_imputed_tau_PET (tau_synthesis_train.py:get_unet).

Original architecture: Kolařík, M., Burget, R., Uher, V., Říha, K., & Dutta, M. K.
(2019). "Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine
Segmentation." Applied Sciences 9(3). https://github.com/mrkolarik/3D-brain-segmentation
Adapted for FDG→tau PET synthesis by J. Lee.

This is a deterministic image-to-image regressor: single input channel → single
output channel, linear final activation, trained with MSE. The dense skip pattern
(each level concatenates the level input onto each conv output) is reproduced
exactly, including the upsampling skips, which use the *first* conv of each encoder
level (conv32/conv22/conv12) rather than the concatenated tensor — matching the
upstream Keras code.

Channel arithmetic is hard-coded to mirror the original verbatim; see the inline
comments for the running channel count at each tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_ch, out_ch):
    """3x3x3 same-padding conv (ReLU applied in forward)."""
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)


class DenseUNet3D(nn.Module):
    """Faithful PyTorch port of the upstream 3D Dense-U-Net regressor.

    Input/output: (N, in_ch, H, W, D) → (N, out_ch, H, W, D).
    H, W, D must each be divisible by 16 (four 2x downsampling levels).
    The lab volume shape (96, 112, 96) satisfies this (96/16=6, 112/16=7).
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(2)

        c0 = in_ch
        # ── Encoder ───────────────────────────────────────────────────────────
        self.conv11 = _conv(c0, 32)                 # → 32
        self.conv12 = _conv(c0 + 32, 32)            # conc11=c0+32 → 32
        p1 = c0 + 32                                # pool1 = cat(in, conv12)

        self.conv21 = _conv(p1, 64)                 # → 64
        self.conv22 = _conv(p1 + 64, 64)            # conc21=p1+64 → 64
        p2 = p1 + 64                                # pool2 = cat(pool1, conv22)

        self.conv31 = _conv(p2, 128)                # → 128
        self.conv32 = _conv(p2 + 128, 128)          # conc31=p2+128 → 128
        p3 = p2 + 128                               # pool3 = cat(pool2, conv32)

        self.conv41 = _conv(p3, 256)                # → 256
        self.conv42 = _conv(p3 + 256, 256)          # conc41=p3+256 → 256
        p4 = p3 + 256                               # pool4 = cat(pool3, conv42)

        self.conv51 = _conv(p4, 512)                # → 512
        self.conv52 = _conv(p4 + 512, 512)          # conc51=p4+512 → 512
        c52 = p4 + 512                              # conc52 = cat(pool4, conv52)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up6 = nn.ConvTranspose3d(c52, 256, kernel_size=2, stride=2)
        u6 = 256 + p4                               # cat(up6, conc42); conc42 channels = p4
        self.conv61 = _conv(u6, 256)
        self.conv62 = _conv(u6 + 256, 256)          # conc61=u6+256 → 256
        c62 = u6 + 256                              # conc62 = cat(up6, conv62)

        self.up7 = nn.ConvTranspose3d(c62, 128, kernel_size=2, stride=2)
        u7 = 128 + 128                              # cat(up7, conv32);  conv32 channels = 128
        self.conv71 = _conv(u7, 128)
        self.conv72 = _conv(u7 + 128, 128)
        c72 = u7 + 128                              # conc72 = cat(up7, conv72)

        self.up8 = nn.ConvTranspose3d(c72, 64, kernel_size=2, stride=2)
        u8 = 64 + 64                                # cat(up8, conv22);  conv22 channels = 64
        self.conv81 = _conv(u8, 64)
        self.conv82 = _conv(u8 + 64, 64)
        c82 = u8 + 64                               # conc82 = cat(up8, conv82)

        self.up9 = nn.ConvTranspose3d(c82, 32, kernel_size=2, stride=2)
        u9 = 32 + 32                                # cat(up9, conv12);  conv12 channels = 32
        self.conv91 = _conv(u9, 32)
        self.conv92 = _conv(u9 + 32, 32)
        c92 = u9 + 32                               # conc92 = cat(up9, conv92)

        self.out = nn.Conv3d(c92, out_ch, kernel_size=1)  # linear output

    def forward(self, x):
        r = self.relu
        inp = x

        conv11 = r(self.conv11(inp))
        conc11 = torch.cat([inp, conv11], dim=1)
        conv12 = r(self.conv12(conc11))
        conc12 = torch.cat([inp, conv12], dim=1)
        pool1 = self.pool(conc12)

        conv21 = r(self.conv21(pool1))
        conc21 = torch.cat([pool1, conv21], dim=1)
        conv22 = r(self.conv22(conc21))
        conc22 = torch.cat([pool1, conv22], dim=1)
        pool2 = self.pool(conc22)

        conv31 = r(self.conv31(pool2))
        conc31 = torch.cat([pool2, conv31], dim=1)
        conv32 = r(self.conv32(conc31))
        conc32 = torch.cat([pool2, conv32], dim=1)
        pool3 = self.pool(conc32)

        conv41 = r(self.conv41(pool3))
        conc41 = torch.cat([pool3, conv41], dim=1)
        conv42 = r(self.conv42(conc41))
        conc42 = torch.cat([pool3, conv42], dim=1)
        pool4 = self.pool(conc42)

        conv51 = r(self.conv51(pool4))
        conc51 = torch.cat([pool4, conv51], dim=1)
        conv52 = r(self.conv52(conc51))
        conc52 = torch.cat([pool4, conv52], dim=1)

        up6 = torch.cat([self.up6(conc52), conc42], dim=1)
        conv61 = r(self.conv61(up6))
        conc61 = torch.cat([up6, conv61], dim=1)
        conv62 = r(self.conv62(conc61))
        conc62 = torch.cat([up6, conv62], dim=1)

        up7 = torch.cat([self.up7(conc62), conv32], dim=1)
        conv71 = r(self.conv71(up7))
        conc71 = torch.cat([up7, conv71], dim=1)
        conv72 = r(self.conv72(conc71))
        conc72 = torch.cat([up7, conv72], dim=1)

        up8 = torch.cat([self.up8(conc72), conv22], dim=1)
        conv81 = r(self.conv81(up8))
        conc81 = torch.cat([up8, conv81], dim=1)
        conv82 = r(self.conv82(conc81))
        conc82 = torch.cat([up8, conv82], dim=1)

        up9 = torch.cat([self.up9(conc82), conv12], dim=1)
        conv91 = r(self.conv91(up9))
        conc91 = torch.cat([up9, conv91], dim=1)
        conv92 = r(self.conv92(conc91))
        conc92 = torch.cat([up9, conv92], dim=1)

        return self.out(conc92)  # linear


if __name__ == "__main__":
    # Shape sanity check at the lab volume resolution.
    m = DenseUNet3D(in_ch=1, out_ch=1)
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.randn(1, 1, 96, 112, 96)
    with torch.no_grad():
        y = m(x)
    print(f"params: {n_params/1e6:.2f}M   in {tuple(x.shape)} → out {tuple(y.shape)}")
    assert y.shape == x.shape
