"""
3D pix2pix components used by Ea-GAN (Yu et al., IEEE TMI 2019; official code
github.com/by-lab/Ea-GANs, MIT license, itself based on pytorch-CycleGAN-and-pix2pix).

  UNet3DGenerator     the `unet_128` generator: 7 stride-2 (k4,p1) down-samplings so a 128³
                      input reaches a 1³ bottleneck, ngf=64 → 64/128/256/512(×4) channels,
                      LeakyReLU(0.2) encoder / ReLU decoder, BatchNorm3d, dropout 0.5 in the
                      innermost ngf*8 blocks (--use_dropout), Tanh output. num_downs is a
                      parameter so smaller volumes can use fewer levels.
  PatchGAN3D          NLayerDiscriminator (basic: n_layers=3, ndf=64, k4): 3D 70³-style PatchGAN.
  sobel3d             the edge-map layer of Ea-GAN: three 3×3×3 Sobel kernels (one per axis),
                      input padded with -1 (background of a [-1,1] image), response /4, L2 norm
                      over the three directions, then tanh(·)·2-1 → edge map in [-1,1].
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm3d(kind):
    if kind == "batch":
        return lambda c: nn.BatchNorm3d(c, affine=True, track_running_stats=True)
    if kind == "instance":
        return lambda c: nn.InstanceNorm3d(c, affine=True, track_running_stats=False)
    raise ValueError(kind)


class _UnetBlock(nn.Module):
    """One U-Net level (recursive), pix2pix `UnetSkipConnectionBlock` layout."""

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
                 innermost=False, norm="batch", use_dropout=False):
        super().__init__()
        self.outermost = outermost
        N = _norm3d(norm)
        input_nc = outer_nc if input_nc is None else input_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downrelu, downconv, uprelu, upconv, N(outer_nc)]
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downrelu, downconv, N(inner_nc), submodule, uprelu, upconv, N(outer_nc)]
            if use_dropout:
                model.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        y = self.model(x)
        return y if self.outermost else torch.cat([y, x], 1)


class UNet3DGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, num_downs=7, ngf=64, norm="batch", use_dropout=True):
        super().__init__()
        block = _UnetBlock(ngf * 8, ngf * 8, innermost=True, norm=norm)
        for _ in range(num_downs - 5):
            block = _UnetBlock(ngf * 8, ngf * 8, submodule=block, norm=norm, use_dropout=use_dropout)
        block = _UnetBlock(ngf * 4, ngf * 8, submodule=block, norm=norm)
        block = _UnetBlock(ngf * 2, ngf * 4, submodule=block, norm=norm)
        block = _UnetBlock(ngf, ngf * 2, submodule=block, norm=norm)
        self.model = _UnetBlock(out_ch, ngf, input_nc=in_ch, submodule=block, outermost=True, norm=norm)
        self.num_downs = num_downs

    def forward(self, x):
        return self.model(x)


class PatchGAN3D(nn.Module):
    def __init__(self, in_ch, ndf=64, n_layers=3, norm="batch"):
        super().__init__()
        N = _norm3d(norm)
        kw, padw = 4, 1
        seq = [nn.Conv3d(in_ch, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf, nf_prev = 1, 1
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(2 ** n, 8)
            seq += [nn.Conv3d(ndf * nf_prev, ndf * nf, kw, stride=2, padding=padw),
                    N(ndf * nf), nn.LeakyReLU(0.2, True)]
        nf_prev, nf = nf, min(2 ** n_layers, 8)
        seq += [nn.Conv3d(ndf * nf_prev, ndf * nf, kw, stride=1, padding=padw),
                N(ndf * nf), nn.LeakyReLU(0.2, True),
                nn.Conv3d(ndf * nf, 1, kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


def _sobel_kernels():
    a = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
    z = np.zeros((3, 3))
    k = np.zeros((3, 1, 3, 3, 3), dtype=np.float32)
    k[0, 0, 0], k[0, 0, 1], k[0, 0, 2] = a, z, -a          # gradient along dim 0
    k[1, 0, :, 0], k[1, 0, :, 1], k[1, 0, :, 2] = a, z, -a  # along dim 1
    k[2, 0, :, :, 0], k[2, 0, :, :, 1], k[2, 0, :, :, 2] = a, z, -a  # along dim 2
    return torch.from_numpy(k)


_SOBEL = {}


def sobel3d(x, pad_value=-1.0):
    """Ea-GAN edge map of a (N,1,H,W,D) image in [-1,1] → (N,1,H,W,D) in [-1,1]."""
    key = (x.device, x.dtype)
    if key not in _SOBEL:
        _SOBEL[key] = _sobel_kernels().to(device=x.device, dtype=x.dtype)
    k = _SOBEL[key]
    xp = F.pad(x, (1, 1, 1, 1, 1, 1), mode="constant", value=pad_value)
    g = F.conv3d(xp, k) / 4.0                    # (N,3,H,W,D)
    mag = torch.norm(g, p=2, dim=1, keepdim=True)  # (N,1,H,W,D)  (== norm/c*3 with c=3)
    return torch.tanh(mag) * 2.0 - 1.0
