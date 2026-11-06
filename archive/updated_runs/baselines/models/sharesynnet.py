"""
Joint synthesis + diagnosis framework of Wang et al., "Joint learning framework of
cross-modal synthesis and diagnosis for Alzheimer's disease by mining underlying shared
modality information", Medical Image Analysis 91:103032 (2024); reference code
github.com/thibault-wch/Joint-Learning-for-Alzheimer-disease (no license file — the
architectures below are re-implemented from the paper and the reference layer layout).

  ShareSynNet   one 3D encoder-decoder that converts in BOTH directions (MRI→PET with
                alpha=0, PET→MRI with alpha=1). Strided-conv encoder 1→32→128→256→512
                (four stride-2 stages, the first one on the raw input), a bottleneck with
                8-head global self-attention + conv, transposed-conv decoder with skip
                concatenation (512→256→128→32), a final ConvTranspose 32→4 and a 3×3×3 conv
                to 1 channel + Tanh. Every stage is Conv → InstanceNorm → AdaIN → LeakyReLU:
                the AdaIN statistics come from a learned, input-independent "shared code"
                (MLP on a constant vector) and are blended with weight alpha, so alpha
                selects the modality direction.
  DefinedDiscriminator   6-block conv discriminator (32,128,256,512,512,512; five stride-2),
                InstanceNorm, LeakyReLU(0.2), 1-channel patch output. Two are used (D_A on
                PET, D_B on MRI). LSGAN objective.
  AwareNet      the interpretable 2D-convolution diagnosis network: a shared 2D ResNet-18
                stem/two stages applied to every axial slice (TimeDistributed), a slice-
                attention module (per-slice local attention map → grouped conv keys /
                1×1 query → softmax over slices, scaled by 32), a fusion stage with two
                temporal-shift (slice-shift, fold 1/8) + ResNet stages, global average
                pooling, a per-slice classifier averaged over slices → logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────── ShareSynNet (generator) ─────────────────────────────

class SharedCode(nn.Module):
    """Input-independent 512-d code: fc(1024→512) then 3 × [fc(512→512), LeakyReLU]."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.rest = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True))

    def forward(self, batch, device):
        ones = torch.ones(batch, 1024, device=device, dtype=self.fc1.weight.dtype)
        return self.rest(self.fc1(ones))


class AdaIN(nn.Module):
    """x_out = alpha * (sigma·x + mu) + (1-alpha) * x, then LeakyReLU; mu/sigma per channel
    from the shared code through a per-block MLP (sigma passed through ReLU)."""

    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(0.2, True))
        self.mu = nn.Linear(512, ch)
        self.sigma = nn.Linear(512, ch)

    def forward(self, x, code, alpha):
        h = self.fc(code)
        mu = self.mu(h).view(x.shape[0], -1, 1, 1, 1)
        sigma = F.relu(self.sigma(h)).view(x.shape[0], -1, 1, 1, 1)
        y = sigma * x + mu
        return F.leaky_relu(alpha * y + (1.0 - alpha) * x, 0.2)


class _ConvIN(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm3d(cout)

    def forward(self, x):
        return self.norm(self.conv(x))


class _Down(nn.Module):
    def __init__(self, cin, cout, stride=2):
        super().__init__()
        self.conv, self.adain = _ConvIN(cin, cout, stride), AdaIN(cout)

    def forward(self, x, code, alpha):
        return self.adain(self.conv(x), code, alpha)


class GlobalAttention3D(nn.Module):
    """Multi-head (8) self-attention over all voxels of a bottleneck feature map, with
    L2-normalised queries/keys, residual output."""

    def __init__(self, ch, heads=8):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv3d(ch, ch * 3, 1, bias=False)

    def forward(self, x):
        b, c, d, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        rs = lambda t: t.reshape(b, self.heads, c // self.heads, d * h * w).transpose(-2, -1)  # noqa: E731
        q, k, v = rs(q), rs(k), rs(v)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        out = (attn @ v).transpose(-2, -1).reshape(b, c, d, h, w)
        return out + x


class _Bottleneck(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.attn, self.conv, self.adain = GlobalAttention3D(ch, 8), _ConvIN(ch, ch, 1), AdaIN(ch)

    def forward(self, x, code, alpha):
        return self.adain(self.conv(self.attn(x)), code, alpha)


class _Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, 4, stride=2, padding=1)
        self.conv, self.adain = _ConvIN(cout * 2, cout, 1), AdaIN(cout)

    def forward(self, x, skip, code, alpha):
        x = torch.cat([self.up(x), skip], 1)
        return self.adain(self.conv(x), code, alpha)


class ShareSynNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.code = SharedCode()
        self.down1, self.down2 = _Down(in_ch, 32), _Down(32, 128)
        self.down3, self.down4 = _Down(128, 256), _Down(256, 512)
        self.bottleneck = _Bottleneck(512)
        self.up4, self.up3, self.up2 = _Up(512, 256), _Up(256, 128), _Up(128, 32)
        self.up1 = nn.ConvTranspose3d(32, 4, 4, stride=2, padding=1)
        self.out = nn.Sequential(nn.Conv3d(4, out_ch, 3, padding=1), nn.Tanh())

    def forward(self, x, alpha: float):
        code = self.code(x.shape[0], x.device)
        x1 = self.down1(x, code, alpha)
        x2 = self.down2(x1, code, alpha)
        x3 = self.down3(x2, code, alpha)
        x4 = self.down4(x3, code, alpha)
        b = self.bottleneck(x4, code, alpha)
        y = self.up4(b, x3, code, alpha)
        y = self.up3(y, x2, code, alpha)
        y = self.up2(y, x1, code, alpha)
        return self.out(self.up1(y))


# ───────────────────────────── Discriminator ─────────────────────────────

class DefinedDiscriminator(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        def cbr(cin, cout, stride):
            return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1),
                                 nn.InstanceNorm3d(cout), nn.LeakyReLU(0.2, True))
        self.model = nn.Sequential(cbr(in_ch, 32, 2), cbr(32, 128, 2), cbr(128, 256, 2),
                                   cbr(256, 512, 2), cbr(512, 512, 2), cbr(512, 512, 1),
                                   nn.Conv3d(512, 1, 3, padding=1))

    def forward(self, x):
        return self.model(x)


# ───────────────────────────── AwareNet (diagnosis) ─────────────────────────────

class BasicBlock2D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout),
                               nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                               nn.BatchNorm2d(cout))
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.f(x) + self.shortcut(x))


def _stage(cin, cout, n_blocks, stride):
    layers, c = [], cin
    for s in [stride] + [1] * (n_blocks - 1):
        layers.append(BasicBlock2D(c, cout, s))
        c = cout
    return nn.Sequential(*layers)


def _time_distributed(module, x):
    """Apply a 2D module to every slice of x (B, S, C, H, W)."""
    b, s = x.shape[:2]
    y = module(x.reshape(b * s, *x.shape[2:]))
    return y.reshape(b, s, *y.shape[1:])


def _temporal_shift(x, fold_div=8):
    c = x.shape[2]
    fold = c // fold_div
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]
    out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
    return out


class SliceAttention(nn.Module):
    def __init__(self, n_slices, ch):
        super().__init__()
        self.local = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.k = nn.Conv2d(n_slices, n_slices, 3, padding=1, groups=n_slices, bias=False)
        self.q = nn.Conv2d(n_slices, 1, 1, bias=False)
        self.scale = 32.0

    def forward(self, x):                       # x: (B, S, C, H, W)
        b, s = x.shape[:2]
        loc = _time_distributed(self.local, x).squeeze(2)   # (B, S, H, W)
        k = self.k(loc).reshape(b, s, -1)
        q = self.q(loc).reshape(b, 1, -1)
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)).squeeze(1) * self.scale
        w = attn.softmax(dim=-1)                # (B, S)
        return x * w.view(b, s, 1, 1, 1) + x, w


class AwareNet(nn.Module):
    """Input: (B, 1, X, Y, Z) volume; slices are taken along the LAST axis (axial)."""

    def __init__(self, num_classes=2, n_slices=96, base=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, base, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(base),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1),
                                  _stage(base, base, 2, 1), _stage(base, base * 2, 2, 2))
        self.attn = SliceAttention(n_slices, base * 2)
        self.fuse1 = _stage(base * 2, base * 4, 2, 2)
        self.fuse2 = _stage(base * 4, base * 8, 3, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(base * 8, base * 2), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                        nn.Linear(base * 2, num_classes))

    def forward(self, vol):
        x = vol.squeeze(1).permute(0, 3, 1, 2).unsqueeze(2)    # (B, Z, 1, X, Y): one 2D slice per z
        x = _time_distributed(self.stem, x)
        x, slice_w = self.attn(x)
        x = _time_distributed(self.fuse1, _temporal_shift(x))
        x = _time_distributed(self.fuse2, _temporal_shift(x))
        x = _time_distributed(self.pool, x).flatten(2)          # (B, Z, C)
        logits = self.classifier(x).mean(dim=1)                 # average the per-slice logits
        return logits, slice_w
