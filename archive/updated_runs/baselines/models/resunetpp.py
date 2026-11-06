"""
2D ResUNet++ generator + 2D PatchGAN discriminator of the CT → PET conditional GAN.

Source: Salehjahromi et al., "Synthetic PET from CT improves diagnosis and prognosis for
lung cancer: Proof of concept", Cell Reports Medicine 5:101463 (2024); official code
github.com/WuLabMDA/Synthetic-PET-from-CT (pix2pix framework; the released checkpoint's
train_opt.txt records the configuration reproduced here).

The model is 2.5D: the generator takes a stack of 7 adjacent axial slices of the source
modality (input_nc=7) and predicts the 3 central slices of the target (output_nc=3).
ResUNet++ (Jha et al. 2019) with filters [72,120,200,330,512] + a 16-channel head, ReLU
output, BatchNorm; squeeze-excite blocks, ASPP bridge/head, attention gates; three
stride-2 stages (input dims must be divisible by 8).

Discriminator: pix2pix `basic` 70×70 PatchGAN (n_layers=3, ndf=64, BatchNorm) applied to
cat(3 central source slices, 3 target slices) = 6 channels; vanilla (BCE) GAN loss.
"""
import torch
import torch.nn as nn


def _norm2d(kind):
    if kind == "batch":
        return lambda c: nn.BatchNorm2d(c)
    if kind == "instance":
        return lambda c: nn.InstanceNorm2d(c, affine=False, track_running_stats=False)
    raise ValueError(kind)


class ResidualConv(nn.Module):
    def __init__(self, cin, cout, stride, N):
        super().__init__()
        self.block = nn.Sequential(N(cin), nn.ReLU(), nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                                   N(cout), nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), N(cout))

    def forward(self, x):
        return self.block(x) + self.skip(x)


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c, c // r, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(c // r, c, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)


class ASPP(nn.Module):
    def __init__(self, cin, cout, N, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin, cout, 3, padding=r, dilation=r), nn.ReLU(inplace=True), N(cout))
            for r in rates])
        self.out = nn.Conv2d(len(rates) * cout, cout, 1)

    def forward(self, x):
        return self.out(torch.cat([b(x) for b in self.branches], 1))


class AttentionGate(nn.Module):
    def __init__(self, c_enc, c_dec, cout, N):
        super().__init__()
        self.enc = nn.Sequential(N(c_enc), nn.ReLU(), nn.Conv2d(c_enc, cout, 3, padding=1), nn.MaxPool2d(2, 2))
        self.dec = nn.Sequential(N(c_dec), nn.ReLU(), nn.Conv2d(c_dec, cout, 3, padding=1))
        self.attn = nn.Sequential(N(cout), nn.ReLU(), nn.Conv2d(cout, 1, 1))

    def forward(self, x_enc, x_dec):
        return self.attn(self.enc(x_enc) + self.dec(x_dec)) * x_dec


class ResUNetPlusPlus(nn.Module):
    def __init__(self, in_ch=7, out_ch=3, filters=(72, 120, 200, 330, 512, 16), norm="batch",
                 final_act="relu"):
        super().__init__()
        N = _norm2d(norm)
        f = filters
        self.input_layer = nn.Sequential(nn.Conv2d(in_ch, f[0], 7, padding=3), N(f[0]), nn.ReLU(),
                                         nn.Conv2d(f[0], f[0], 7, padding=3))
        self.input_skip = nn.Conv2d(in_ch, f[0], 7, padding=3)
        self.se1, self.rc1 = SqueezeExcite(f[0]), ResidualConv(f[0], f[1], 2, N)
        self.se2, self.rc2 = SqueezeExcite(f[1]), ResidualConv(f[1], f[2], 2, N)
        self.se3, self.rc3 = SqueezeExcite(f[2]), ResidualConv(f[2], f[3], 2, N)
        self.bridge = ASPP(f[3], f[4], N)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.att1, self.urc1 = AttentionGate(f[2], f[4], f[4], N), ResidualConv(f[4] + f[2], f[3], 1, N)
        self.att2, self.urc2 = AttentionGate(f[1], f[3], f[3], N), ResidualConv(f[3] + f[1], f[2], 1, N)
        self.att3, self.urc3 = AttentionGate(f[0], f[2], f[2], N), ResidualConv(f[2] + f[0], f[1], 1, N)
        self.aspp_out = ASPP(f[1], f[0], N)
        head = [nn.Conv2d(f[0], f[5], 1), nn.Conv2d(f[5], out_ch, 1)]
        if final_act == "relu":
            head.append(nn.ReLU())
        elif final_act == "tanh":
            head.append(nn.Tanh())
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.rc1(self.se1(x1))
        x3 = self.rc2(self.se2(x2))
        x4 = self.rc3(self.se3(x3))
        x5 = self.bridge(x4)
        d = self.urc1(torch.cat([self.up(self.att1(x3, x5)), x3], 1))
        d = self.urc2(torch.cat([self.up(self.att2(x2, d)), x2], 1))
        d = self.urc3(torch.cat([self.up(self.att3(x1, d)), x1], 1))
        return self.head(self.aspp_out(d))


class PatchGAN2D(nn.Module):
    def __init__(self, in_ch, ndf=64, n_layers=3, norm="batch"):
        super().__init__()
        N = _norm2d(norm)
        kw, padw = 4, 1
        seq = [nn.Conv2d(in_ch, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf, nf_prev = 1, 1
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * nf_prev, ndf * nf, kw, stride=2, padding=padw), N(ndf * nf),
                    nn.LeakyReLU(0.2, True)]
        nf_prev, nf = nf, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * nf_prev, ndf * nf, kw, stride=1, padding=padw), N(ndf * nf),
                nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * nf, 1, kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)
