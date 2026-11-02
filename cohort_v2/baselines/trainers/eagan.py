"""
Ea-GAN trainer (Yu et al., IEEE TMI 2019) — gEa-GAN and dEa-GAN, following the official
train_gEaGAN.sh / train_dEaGAN.sh options:

  generator unet_128 (3D pix2pix U-Net, 7 levels, ngf 64, BatchNorm, dropout), PatchGAN
  discriminator (basic, ndf 64), LSGAN objective with label smoothing (--labelSmooth),
  L1 weight lambda_A = 300, Sobel edge-L1 weight lambda_sobel = 100 ramped linearly from
  0 over the first 20 epochs (--rise_sobelLoss), Adam lr 2e-4 beta1 0.5 for G and D,
  100 epochs at constant lr + 50 epochs of linear decay, image pool of 4, random flips.
  gEa-GAN: edge loss in the generator only; D sees cat(source, target).
  dEa-GAN: additionally the discriminator sees cat(source, target, sobel(target)).

Adaptations for the lab data (see docs/BASELINES.md):
  * volumes are (96,112,96); the 7-level U-Net needs 128³, so inputs are padded to
    --pad-shape (default 128,128,128) with the background value (-1) and cropped back.
  * data are mapped [0,1] → [-1,1] (Tanh output); the generator output is masked to the
    DK86 region (background := -1) before the losses and the discriminator, exactly as
    generate.py masks predictions.
  * paper batch size 6 → default 2 (memory at 128³); flips only left-right (MNI x).
  * the official dEa-GAN code passes REAL pairs through the fake-image pool too; that
    (a bug) is not reproduced — only fakes are pooled.
"""
import torch

from .base import BaseTrainer, linear_decay_lambda
from ..models.common import GANLoss, ImagePool, crop_pads, from_pm1, init_weights, pad_to, to_pm1
from ..models.pix2pix3d import PatchGAN3D, UNet3DGenerator, sobel3d


class EaGANTrainer(BaseTrainer):
    name = "eagan"

    def build(self):
        a = self.args
        self.pad_shape = tuple(int(v) for v in a.pad_shape.split(","))
        G = UNet3DGenerator(1, 1, num_downs=a.num_downs, ngf=a.ngf, norm=a.gan_norm, use_dropout=a.use_dropout)
        d_in = 3 if a.eagan_variant == "d" else 2
        D = PatchGAN3D(d_in, ndf=a.ndf, n_layers=3, norm=a.gan_norm)
        init_weights(G, "normal", 0.02)
        init_weights(D, "normal", 0.02)
        self.nets = {"G": G.to(self.device), "D": D.to(self.device)}
        self.optimizers = {
            "G": torch.optim.Adam(G.parameters(), lr=a.lr, betas=(a.beta1, 0.999)),
            "D": torch.optim.Adam(D.parameters(), lr=a.lr, betas=(a.beta1, 0.999))}
        f = linear_decay_lambda(a.n_const_epochs, max(a.epochs - a.n_const_epochs, 1))
        self.scheds = [torch.optim.lr_scheduler.LambdaLR(o, f) for o in self.optimizers.values()]
        self.gan = GANLoss(a.gan_mode, label_smooth=a.label_smooth)
        self.pool = ImagePool(a.pool_size)
        self.sobel_lambda = 0.0 if a.rise_sobel else a.lambda_sobel
        self._maskp = None

    def _prep(self, mri, pet):
        """[0,1] volumes → padded [-1,1] volumes + padded mask."""
        x, pads = pad_to(to_pm1(mri), self.pad_shape, value=-1.0)
        y, _ = pad_to(to_pm1(pet), self.pad_shape, value=-1.0)
        if self._maskp is None or self._maskp.shape[2:] != x.shape[2:]:
            self._maskp, _ = pad_to(self.mask, self.pad_shape, value=0.0)
        return x, y, pads

    def _mask_fake(self, fake):
        return fake * self._maskp + (-1.0) * (1.0 - self._maskp)

    def _d_input(self, x, y, edge):
        return torch.cat([x, y, edge], 1) if self.args.eagan_variant == "d" else torch.cat([x, y], 1)

    def train_epoch(self, loader, epoch):
        a = self.args
        G, D = self.nets["G"], self.nets["D"]
        oG, oD = self.optimizers["G"], self.optimizers["D"]
        if a.rise_sobel and epoch <= 20:
            self.sobel_lambda = a.lambda_sobel / 20.0 * epoch
        sums = {"G_gan": 0.0, "G_l1": 0.0, "G_sobel": 0.0, "D": 0.0}
        n = 0
        for pet, mri, _ in loader:
            pet, mri = pet.to(self.device), mri.to(self.device)
            if a.flip_lr and torch.rand(1).item() < 0.5:
                pet, mri = pet.flip(2), mri.flip(2)
            x, y, _ = self._prep(mri, pet)
            fake = self._mask_fake(G(x))
            real_edge = sobel3d(y).detach()
            fake_edge = sobel3d(fake)
            # ── D ──
            oD.zero_grad(set_to_none=True)
            fake_in = self.pool.query(self._d_input(x, fake, fake_edge).detach())
            loss_d = 0.5 * (self.gan(D(fake_in), False) + self.gan(D(self._d_input(x, y, real_edge)), True))
            loss_d.backward()
            oD.step()
            # ── G ──
            oG.zero_grad(set_to_none=True)
            for p in D.parameters():
                p.requires_grad_(False)
            g_gan = self.gan(D(self._d_input(x, fake, fake_edge)), True)
            if a.pixel_loss_masked:
                m = self._maskp
                g_l1 = ((fake - y).abs() * m).sum() / (m.sum() * fake.shape[0])
                g_sob = ((fake_edge - real_edge).abs() * m).sum() / (m.sum() * fake.shape[0])
            else:
                g_l1 = (fake - y).abs().mean()
                g_sob = (fake_edge - real_edge).abs().mean()
            loss_g = g_gan + a.lambda_l1 * g_l1 + self.sobel_lambda * g_sob
            loss_g.backward()
            oG.step()
            for p in D.parameters():
                p.requires_grad_(True)
            b = pet.shape[0]
            sums["G_gan"] += g_gan.item() * b; sums["G_l1"] += g_l1.item() * b
            sums["G_sobel"] += g_sob.item() * b; sums["D"] += loss_d.item() * b
            n += b
        out = {k: v / max(n, 1) for k, v in sums.items()}
        out["sobel_lambda"] = self.sobel_lambda
        out["lr"] = oG.param_groups[0]["lr"]
        return out

    def on_epoch_end(self, epoch):
        for s in self.scheds:
            s.step()

    @torch.no_grad()
    def predict(self, mri):
        x, pads = pad_to(to_pm1(mri), self.pad_shape, value=-1.0)
        if self._maskp is None or self._maskp.shape[2:] != x.shape[2:]:
            self._maskp, _ = pad_to(self.mask, self.pad_shape, value=0.0)
        fake = self._mask_fake(self.nets["G"](x))
        return from_pm1(crop_pads(fake, pads)) * self.mask
