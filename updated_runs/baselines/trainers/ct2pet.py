"""
2.5D conditional GAN of Salehjahromi et al. (Cell Rep Med 2024), "CT→PET", applied to
MRI → tau PET. Configuration from the released checkpoint's train_opt.txt:

  netG ResUnetPlusPlus (input_nc 7 adjacent slices → output_nc 3 central slices, ReLU
  output, BatchNorm), netD basic PatchGAN (n_layers 3, ndf 64) on the 3 central source
  slices + 3 target slices, gan_mode vanilla, lambda_L1 6000 (README quotes 4000), Adam
  lr 2e-4 beta1 0.5, batch 5, 100 epochs constant lr + 100 linear decay, no image pool,
  random affine augmentation (p=0.95: rotation ±45°, translation 10 %, scale 0.85–1.15,
  shear ±7°; else rotation ±180°) applied identically to source and target, target
  intensities gamma-compressed (sqrt) before training and squared back at inference.

Adaptations: axial slices of the (96,112,96) MNI volume (96 slabs per subject, all of
them each epoch — the original thinned its 1,478-patient CT set to every 3rd slab);
the volume is zero-padded by 3 slices at both ends so every slice gets a prediction
(the original leaves the first/last 3 slices empty); output masked to DK86.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from .base import BaseTrainer, linear_decay_lambda
from ..models.common import GANLoss, init_weights
from ..models.resunetpp import PatchGAN2D, ResUNetPlusPlus


class SlabDataset(Dataset):
    """(mri 7-slab, pet 3-slab) samples along the last (axial) axis of cached volumes."""

    def __init__(self, cached, n_in=7, n_out=3, augment=True, gamma=0.5, stride=1):
        self.c, self.n_in, self.n_out, self.augment, self.gamma = cached, n_in, n_out, augment, gamma
        self.half = n_in // 2
        D = cached[0][0].shape[-1]
        self.D = D
        self.index = [(s, k) for s in range(len(cached)) for k in range(0, D, stride)]

    def __len__(self):
        return len(self.index)

    @staticmethod
    def pad_z(vol, half):
        return F.pad(vol, (half, half), mode="constant", value=0.0)    # pad last dim only

    def __getitem__(self, i):
        s, k = self.index[i]
        pet, mri, _ = self.c[s]
        mri_p, pet_p = self.pad_z(mri, self.half), self.pad_z(pet, self.half)
        kk = k + self.half
        x = mri_p[0, :, :, kk - self.half: kk + self.half + 1].permute(2, 0, 1).contiguous()  # (7,H,W)
        ho = self.n_out // 2
        y = pet_p[0, :, :, kk - ho: kk + ho + 1].permute(2, 0, 1).contiguous()                 # (3,H,W)
        if self.gamma and self.gamma != 1.0:
            y = y.clamp(min=0) ** self.gamma
        if self.augment:
            x, y = self.random_affine(x, y)
        return x, y

    @staticmethod
    def random_affine(x, y):
        if torch.rand(1).item() < 0.95:
            ang = float(torch.empty(1).uniform_(-45, 45))
        else:
            ang = float(torch.empty(1).uniform_(-180, 180))
        H, W = x.shape[-2:]
        tx = int(torch.empty(1).uniform_(-0.10 * W, 0.10 * W)); ty = int(torch.empty(1).uniform_(-0.10 * H, 0.10 * H))
        sc = float(torch.empty(1).uniform_(0.85, 1.15))
        sh = float(torch.empty(1).uniform_(-7, 7))
        kw = dict(angle=ang, translate=[tx, ty], scale=sc, shear=[sh, 0.0])
        return TF.affine(x, **kw), TF.affine(y, **kw)


class CT2PETTrainer(BaseTrainer):
    name = "ct2pet"

    def build(self):
        a = self.args
        G = ResUNetPlusPlus(a.n_in_slices, a.n_out_slices, norm=a.gan_norm, final_act="relu")
        D = PatchGAN2D(2 * a.n_out_slices, ndf=a.ndf, n_layers=3, norm=a.gan_norm)
        init_weights(G, "normal", 0.02); init_weights(D, "normal", 0.02)
        self.nets = {"G": G.to(self.device), "D": D.to(self.device)}
        self.optimizers = {"G": torch.optim.Adam(G.parameters(), lr=a.lr, betas=(a.beta1, 0.999)),
                           "D": torch.optim.Adam(D.parameters(), lr=a.lr, betas=(a.beta1, 0.999))}
        f = linear_decay_lambda(a.n_const_epochs, max(a.epochs - a.n_const_epochs, 1))
        self.scheds = [torch.optim.lr_scheduler.LambdaLR(o, f) for o in self.optimizers.values()]
        self.gan = GANLoss(a.gan_mode)
        self.center = a.n_in_slices // 2
        self.ho = a.n_out_slices // 2

    def make_train_loader(self, cached):
        ds = SlabDataset(cached, self.args.n_in_slices, self.args.n_out_slices,
                         augment=self.args.augment, gamma=self.args.pet_gamma, stride=self.args.slab_stride)
        return DataLoader(ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers,
                          pin_memory=True, drop_last=False, persistent_workers=self.args.num_workers > 0)

    def _cond(self, x, y):
        c = self.center
        return torch.cat([x[:, c - self.ho: c + self.ho + 1], y], 1)

    def train_epoch(self, loader, epoch):
        a = self.args
        G, D = self.nets["G"], self.nets["D"]
        oG, oD = self.optimizers["G"], self.optimizers["D"]
        sums = {"G_gan": 0.0, "G_l1": 0.0, "D": 0.0}; n = 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            fake = G(x)
            oD.zero_grad(set_to_none=True)
            loss_d = 0.5 * (self.gan(D(self._cond(x, fake.detach())), False) + self.gan(D(self._cond(x, y)), True))
            loss_d.backward(); oD.step()
            oG.zero_grad(set_to_none=True)
            for p in D.parameters():
                p.requires_grad_(False)
            g_gan = self.gan(D(self._cond(x, fake)), True)
            g_l1 = (fake - y).abs().mean()
            (g_gan + a.lambda_l1 * g_l1).backward(); oG.step()
            for p in D.parameters():
                p.requires_grad_(True)
            b = x.shape[0]
            sums["G_gan"] += g_gan.item() * b; sums["G_l1"] += g_l1.item() * b; sums["D"] += loss_d.item() * b; n += b
        out = {k: v / max(n, 1) for k, v in sums.items()}
        out["lr"] = oG.param_groups[0]["lr"]
        return out

    def on_epoch_end(self, epoch):
        for s in self.scheds:
            s.step()

    @torch.no_grad()
    def predict(self, mri):
        """Slide the 7-slice window over every axial position, keep the centre output slice."""
        G = self.nets["G"]
        a = self.args
        half, c = a.n_in_slices // 2, self.center
        N, _, H, W, D = mri.shape
        outs = []
        for i in range(N):
            vol = SlabDataset.pad_z(mri[i], half)                       # (1,H,W,D+2half)
            slabs = torch.stack([vol[0, :, :, k: k + a.n_in_slices].permute(2, 0, 1) for k in range(D)])  # (D,7,H,W)
            preds = []
            for j in range(0, D, a.infer_batch):
                preds.append(G(slabs[j: j + a.infer_batch])[:, self.ho])   # centre of the 3 output slices
            pv = torch.cat(preds, 0).permute(1, 2, 0)                        # (H,W,D)
            pv = pv.clamp(0.0, 1.0)                                          # their post_gamma_PET clips to [0,1]
            if a.pet_gamma and a.pet_gamma != 1.0:
                pv = pv ** (1.0 / a.pet_gamma)
            outs.append(pv.unsqueeze(0))
        return torch.stack(outs) * self.mask
