#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
train_lp_modality.py – TauGenNet with optional 3D LP and/or Modality-Aware AE loss.

Two auxiliary losses can be added to the AE stage (independently or together):

  --lp-weight   W   3D Laplacian Pyramid loss (multi-scale structural fidelity, no
                    pretrained network needed). Default 0 = disabled.

  --modality-weight W   Modality-Aware loss (CrossEntropy through a frozen 3D ResNet-18
                        pretrained to distinguish brain MRI vs. PET). Forces the AE to
                        reconstruct PET scans that look like PET, not MRI. Default 0 = disabled.

  --modality-ckpt PATH  Path to the modality discriminator checkpoint.
                        Defaults to the MaM-DiT epoch-60 checkpoint.

Either, both, or neither auxiliary loss can be active. All other training logic is
identical to train.py.

Usage
-----
# AE only, LP loss
python scripts/train_lp_modality.py --mode atrophy --diff-epochs 0 --lp-weight 0.1

# AE only, Modality loss
python scripts/train_lp_modality.py --mode atrophy --diff-epochs 0 --modality-weight 0.05

# AE only, both auxiliary losses
python scripts/train_lp_modality.py --mode atrophy --diff-epochs 0 --lp-weight 0.1 --modality-weight 0.05

# Full pipeline (AE with both losses, then diffusion)
python scripts/train_lp_modality.py --mode atrophy --lp-weight 0.1 --modality-weight 0.05

# Skip AE (load checkpoint), train diffusion only
python scripts/train_lp_modality.py --mode atrophy --skip-ae --diff-epochs 2000
"""

import argparse
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.config import (DEVICE, T_STEPS, LR, AE_EPOCHS, DIFF_EPOCHS, BATCH_SIZE,
                        AE_CHECKPOINT_PATH, CHECKPOINT_DIR, FIGURES_DIR, LATENT_CH)
from src import dataset_final     as _dataset_final
from src import dataset_combined  as _dataset_combined
from src.models       import Autoencoder3D, DenoisingUNet3D
from src.diffusion    import DiffusionSchedule
from src.conditioning import build_conditioner
from src.inference    import synthesize_tau_pet


# ── 3D Laplacian Pyramid Loss ─────────────────────────────────────────────────

def laplacian_pyramid_loss_3d(img1, img2, max_levels=4):
    """Multi-scale structural loss on 3D volumes.

    At each level: downsample, upsample back, compute the residual (detail band),
    then accumulate MSE between the two pyramids. Purely self-supervised — no
    pretrained network required.
    """
    def pyramid(x):
        cur, pyr = x, []
        for _ in range(max_levels):
            down = F.avg_pool3d(cur, kernel_size=2, stride=2)
            up   = F.interpolate(down, size=cur.shape[-3:], mode="trilinear",
                                 align_corners=False)
            pyr.append(cur - up)
            cur = down
        return pyr

    return sum(F.mse_loss(a, b) for a, b in zip(pyramid(img1), pyramid(img2)))


# ── Modality-Aware Loss (frozen 3D ResNet-18) ─────────────────────────────────
#
# CrossEntropy(ResNet18_3D(reconstruction), modality_label)
# Labels: 0 = MRI, 1/2/3 = PET task types (this project uses label 1 for tau PET)
#
# The ResNet is frozen: its weights are never updated, but gradients flow through
# it to the AE, telling it "make PET reconstructions that look like PET features."
#
# Inputs are trilinearly resized to 128³ before the forward pass to match the
# spatial_size=128 the network was trained with (your volumes are 96×112×96).

def _conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class _BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1      = _conv3x3x3(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm3d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = _conv3x3x3(planes, planes)
        self.bn2        = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(out + x)


class _ResNet3D(nn.Module):
    """3D ResNet-18 matching the MaM-DiT modality discriminator architecture."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm3d(64)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1   = self._make_layer(64,  2)
        self.layer2   = self._make_layer(128, 2, stride=2)
        self.layer3   = self._make_layer(256, 2, stride=2)
        self.layer4   = self._make_layer(512, 2, stride=2)
        # Hardcoded for spatial_size=128: 128/32 = 4 → AvgPool3d((4,4,4))
        # Inputs are always resized to 128³ before entering this network.
        self.avgpool  = nn.AvgPool3d((4, 4, 4), stride=1)
        self.fc1      = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True),
                                      nn.Dropout(0.5), nn.Linear(256, 128))
        self.fc3      = nn.Linear(128, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * _BasicBlock3D.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes))
        layers = [_BasicBlock3D(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [_BasicBlock3D(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        return self.fc3(self.fc1(x.view(x.size(0), -1)))


class ModalityLoss(nn.Module):
    """Frozen 3D ResNet-18 modality discriminator as a perceptual loss.

    Weights loaded from the MaM-DiT epoch-60 checkpoint. Inputs are trilinearly
    resized to 128³ internally to match the network's expected spatial size.
    """
    DEFAULT_CKPT = ("/scratch/network/sz3962/MaM-DiT/logs/MDM/"
                    "modality_discriminative_model/60_net.pth")

    def __init__(self, ckpt_path=None):
        super().__init__()
        path = ckpt_path or self.DEFAULT_CKPT
        self.net = _ResNet3D(num_classes=4)
        sd = torch.load(path, map_location="cpu")
        self.net.load_state_dict(sd, strict=False)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()  # keep frozen net always in eval regardless of outer .train()
        return self

    def forward(self, x, labels):
        x_r = F.interpolate(x.float(), size=(128, 128, 128),
                            mode="trilinear", align_corners=False)
        return F.cross_entropy(self.net(x_r), labels)


# ── AE loss ───────────────────────────────────────────────────────────────────

def ae_loss(recon, x, mean, logvar,
            kl_weight=1e-4,
            lp_weight=0.0,
            modality_loss_fn=None, modality_label=None, modality_weight=0.0):
    recon_loss = F.l1_loss(recon, x)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    loss       = recon_loss + kl_weight * kl_loss

    if lp_weight > 0.0:
        loss = loss + lp_weight * laplacian_pyramid_loss_3d(recon, x)

    if modality_weight > 0.0 and modality_loss_fn is not None and modality_label is not None:
        loss = loss + modality_weight * modality_loss_fn(recon, modality_label)

    return loss


# ── Latent cache ──────────────────────────────────────────────────────────────

class LatentDataset(Dataset):
    def __init__(self, z_pets, z_mris, conds):
        self.z_pets = z_pets
        self.z_mris = z_mris
        self.conds  = conds

    def __len__(self):
        return len(self.z_pets)

    def __getitem__(self, idx):
        return self.z_pets[idx], self.z_mris[idx], self.conds[idx]


def precompute_latents(ae, dataset, device, desc="Caching latents"):
    ae.eval()
    z_pets, z_mris, conds = [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=desc, leave=False):
            pet, mri, cond = dataset[i]
            z_pets.append(ae.encode_mean(pet.unsqueeze(0).to(device)).cpu().squeeze(0))
            z_mris.append(ae.encode_mean(mri.unsqueeze(0).to(device)).cpu().squeeze(0))
            conds.append(cond)
    return torch.stack(z_pets), torch.stack(z_mris), torch.stack(conds)


# ── AE pretraining ────────────────────────────────────────────────────────────

def train_ae(ae, train_loader, n_epochs, ckpt_path, device,
             val_loader=None, patience=20, val_every=5, figures_dir=None,
             kl_weight=1e-4, lp_weight=0.0,
             modality_loss_fn=None, modality_weight=0.0):
    ae_opt = torch.optim.Adam(ae.parameters(), lr=LR)
    scaler = GradScaler("cuda")
    train_losses             = []
    val_epochs_list          = []
    val_losses_list          = []
    best_val_loss            = float("inf")
    best_epoch               = 0
    epochs_no_improv         = 0
    best_ckpt_path           = ckpt_path.replace(".pt", "_best.pt")

    aux = []
    if lp_weight > 0.0:
        aux.append(f"LP(w={lp_weight})")
    if modality_weight > 0.0:
        aux.append(f"Modality(w={modality_weight})")
    print("=== Autoencoder Pretraining ===", flush=True)
    print(f"    Auxiliary losses: {', '.join(aux) if aux else 'none'}", flush=True)

    start = time.time()
    for epoch in range(n_epochs):
        ae.train()
        epoch_loss = 0.0
        t0 = time.time()

        for pet, mri, _ in train_loader:
            pet, mri = pet.to(device), mri.to(device)
            B = pet.shape[0]
            ae_opt.zero_grad()
            loss = torch.tensor(0.0, device=device)

            with autocast("cuda"):
                # PET reconstruction — modality label 1 (tau PET)
                recon, mean, logvar = ae(pet)
                pet_lbl = torch.ones(B, dtype=torch.long, device=device)
                loss = loss + ae_loss(recon, pet, mean, logvar,
                                      kl_weight=kl_weight,
                                      lp_weight=lp_weight,
                                      modality_loss_fn=modality_loss_fn,
                                      modality_label=pet_lbl,
                                      modality_weight=modality_weight)
                # MRI reconstruction — modality label 0
                recon, mean, logvar = ae(mri)
                mri_lbl = torch.zeros(B, dtype=torch.long, device=device)
                loss = loss + ae_loss(recon, mri, mean, logvar,
                                      kl_weight=kl_weight,
                                      lp_weight=lp_weight,
                                      modality_loss_fn=modality_loss_fn,
                                      modality_label=mri_lbl,
                                      modality_weight=modality_weight)

            scaler.scale(loss).backward()
            scaler.step(ae_opt)
            scaler.update()
            epoch_loss += loss.item()

        avg = epoch_loss / len(train_loader)
        train_losses.append(avg)

        # ── Validation ────────────────────────────────────────────────────────
        val_str = ""
        if val_loader is not None and (epoch + 1) % val_every == 0:
            ae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for pet, mri, _ in val_loader:
                    pet, mri = pet.to(device), mri.to(device)
                    B = pet.shape[0]
                    with autocast("cuda"):
                        recon, mean, logvar = ae(pet)
                        pet_lbl = torch.ones(B, dtype=torch.long, device=device)
                        val_loss += ae_loss(recon, pet, mean, logvar,
                                            kl_weight=kl_weight,
                                            lp_weight=lp_weight,
                                            modality_loss_fn=modality_loss_fn,
                                            modality_label=pet_lbl,
                                            modality_weight=modality_weight).item()
                        recon, mean, logvar = ae(mri)
                        mri_lbl = torch.zeros(B, dtype=torch.long, device=device)
                        val_loss += ae_loss(recon, mri, mean, logvar,
                                            kl_weight=kl_weight,
                                            lp_weight=lp_weight,
                                            modality_loss_fn=modality_loss_fn,
                                            modality_label=mri_lbl,
                                            modality_weight=modality_weight).item()
            val_loss /= len(val_loader)
            val_epochs_list.append(epoch + 1)
            val_losses_list.append(val_loss)
            val_str = f"  val={val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_epoch       = epoch + 1
                epochs_no_improv = 0
                torch.save({"ae": ae.state_dict(), "ae_losses": train_losses,
                            "ae_val_losses": list(zip(val_epochs_list, val_losses_list)),
                            "epoch": epoch + 1, "latent_ch": ae.latent_ch}, best_ckpt_path)
            else:
                epochs_no_improv += val_every
                if epochs_no_improv >= patience:
                    print(f"  Early stop at epoch {epoch+1}: "
                          f"no val improvement for {patience} epochs. "
                          f"Best val={best_val_loss:.4f} at epoch {best_epoch}", flush=True)
                    break

        if (epoch + 1) % 5 == 0:
            elapsed   = time.time() - start
            remaining = (time.time() - t0) * (n_epochs - epoch - 1)
            print(f"  AE {epoch+1:3d}/{n_epochs}  loss={avg:.4f}{val_str}  "
                  f"elapsed={elapsed/60:.1f}m  remaining={remaining/60:.1f}m", flush=True)
            torch.save({"ae": ae.state_dict(), "ae_losses": train_losses,
                        "ae_val_losses": list(zip(val_epochs_list, val_losses_list)),
                        "epoch": epoch + 1, "latent_ch": ae.latent_ch}, ckpt_path)

    print(f"AE pretraining done. Total: {(time.time()-start)/60:.1f}m")

    train_x = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_x, train_losses, label="Train", alpha=0.8)
    if val_epochs_list:
        ax.plot(val_epochs_list, val_losses_list, "o-", lw=1.5, ms=4, label="Val")
        ax.axvline(x=best_epoch, color="g", ls="--", lw=1.5,
                   label=f"Best val (ep {best_epoch})")
    ax.set(xlabel="Epoch", ylabel="Loss", title="AE Loss (LP+Modality)")
    ax.legend()
    fig.tight_layout()
    figs_out = figures_dir or FIGURES_DIR
    os.makedirs(figs_out, exist_ok=True)
    fig.savefig(os.path.join(figs_out, "ae_loss_lp_modality.png"), dpi=150)
    plt.close(fig)
    print(f"AE loss curve saved to {figs_out}/ae_loss_lp_modality.png")

    return train_losses


# ── SSIM monitoring ───────────────────────────────────────────────────────────

def _monitor_ssim(ae, unet, conditioner, schedule, latent_std,
                  val_ds, epoch, monitor_dir, device, n_steps=50):
    unet.eval()
    if hasattr(conditioner, "eval"):
        conditioner.eval()

    pet, mri, cond = val_ds[0]
    real_np = pet.squeeze().numpy()

    gen = synthesize_tau_pet(
        mri.unsqueeze(0), cond, ae, unet, schedule,
        conditioner.encode, latent_std, device=device, n_steps=n_steps,
    )
    gen_np = gen.squeeze().numpy()

    mid      = real_np.shape[2] // 2
    ssim_val = ssim(real_np[:, :, mid], gen_np[:, :, mid], data_range=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(real_np[:, :, mid], cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Real PET"); axes[0].axis("off")
    axes[1].imshow(gen_np[:, :, mid], cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"Generated (ep {epoch})"); axes[1].axis("off")
    axes[2].imshow(np.abs(real_np[:, :, mid] - gen_np[:, :, mid]),
                   cmap="coolwarm", vmin=0, vmax=0.5)
    axes[2].set_title(f"SSIM={ssim_val:.3f}"); axes[2].axis("off")
    plt.tight_layout()

    os.makedirs(monitor_dir, exist_ok=True)
    plt.savefig(os.path.join(monitor_dir, f"epoch_{epoch:04d}.png"),
                dpi=100, bbox_inches="tight")
    plt.close(fig)

    if hasattr(conditioner, "train"):
        conditioner.train()
    unet.train()
    return ssim_val


# ── Diffusion training ────────────────────────────────────────────────────────
# Unchanged from train.py — auxiliary losses only affect the AE stage.

def train_diffusion(ae, unet, conditioner, schedule, latent_std,
                    latent_train_loader, latent_val_loader, val_ds,
                    n_epochs, start_epoch, ckpt_path, device,
                    patience=50, val_every=1, monitor_every=25,
                    monitor_dir=None, figures_dir=None, lr=LR):
    trainable = list(unet.parameters()) + conditioner.trainable_parameters()
    diff_opt  = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    losses          = []
    val_epochs_list = []
    val_losses_list = []
    ae.eval()

    best_val_loss    = float("inf")
    best_epoch       = start_epoch
    epochs_no_improv = 0
    best_ckpt_path   = ckpt_path.replace(".pt", "_best.pt")
    mode_tag         = os.path.basename(ckpt_path).replace("diff_", "").replace(".pt", "")

    print(f"=== Diffusion Training (start_epoch={start_epoch}, patience={patience}) ===",
          flush=True)
    start = time.time()

    for epoch in range(start_epoch, n_epochs):
        unet.train()
        if hasattr(conditioner, "train"):
            conditioner.train()
        epoch_loss = 0.0
        t0 = time.time()

        for z0, zm, cond_data in latent_train_loader:
            z0, zm, cond_data = z0.to(device), zm.to(device), cond_data.to(device)
            B       = z0.shape[0]
            t       = torch.randint(0, T_STEPS, (B,), device=device)
            zt, eps = schedule.q_sample(z0, t)
            ctx     = conditioner.encode(cond_data)
            ht      = torch.cat([zt, zm], dim=1)
            eps_hat = unet(ht, t, ctx)
            loss    = F.mse_loss(eps_hat, eps)

            diff_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            diff_opt.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(latent_train_loader)
        losses.append(avg_train)
        elapsed   = time.time() - start
        remaining = (time.time() - t0) * (n_epochs - epoch - 1)

        val_str = ""
        if (epoch + 1) % val_every == 0:
            unet.eval()
            if hasattr(conditioner, "eval"):
                conditioner.eval()
            val_loss = 0.0
            with torch.no_grad():
                for z0, zm, cond_data in latent_val_loader:
                    z0, zm, cond_data = z0.to(device), zm.to(device), cond_data.to(device)
                    B       = z0.shape[0]
                    t       = torch.randint(0, T_STEPS, (B,), device=device)
                    zt, eps = schedule.q_sample(z0, t)
                    ctx     = conditioner.encode(cond_data)
                    ht      = torch.cat([zt, zm], dim=1)
                    eps_hat = unet(ht, t, ctx)
                    val_loss += F.mse_loss(eps_hat, eps).item()
            val_loss /= len(latent_val_loader)
            val_epochs_list.append(epoch + 1)
            val_losses_list.append(val_loss)
            val_str = f"  val={val_loss:.6f}"

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_epoch       = epoch + 1
                epochs_no_improv = 0
                _save_diff_ckpt(best_ckpt_path, ae, unet, conditioner, latent_std,
                                losses, epoch + 1,
                                diff_val_losses=list(zip(val_epochs_list, val_losses_list)))
            else:
                epochs_no_improv += val_every

        ssim_str = ""
        if (epoch + 1) % monitor_every == 0:
            ssim_val = _monitor_ssim(ae, unet, conditioner, schedule, latent_std,
                                     val_ds, epoch + 1,
                                     monitor_dir or os.path.join(FIGURES_DIR, "monitor"),
                                     device)
            ssim_str = f"  SSIM={ssim_val:.3f}"

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}"
                  f"  train={avg_train:.6f}{val_str}{ssim_str}"
                  f"  elapsed={elapsed/60:.1f}m  remaining={remaining/60:.1f}m",
                  flush=True)
            _save_diff_ckpt(ckpt_path, ae, unet, conditioner, latent_std,
                            losses, epoch + 1,
                            diff_val_losses=list(zip(val_epochs_list, val_losses_list)))

        if epochs_no_improv >= patience:
            print(f"Early stopping at epoch {epoch+1}: "
                  f"no val improvement for {patience} epochs. "
                  f"Best val={best_val_loss:.6f}", flush=True)
            break

    print(f"Diffusion training done. Total: {(time.time()-start)/60:.1f}m")

    train_x = list(range(start_epoch + 1, start_epoch + 1 + len(losses)))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_x, losses, label="Train", alpha=0.8)
    if val_epochs_list:
        ax.plot(val_epochs_list, val_losses_list, "o-", lw=1.5, ms=4, label="Val")
        ax.axvline(x=best_epoch, color="g", ls="--", lw=1.5,
                   label=f"Best val (ep {best_epoch})")
    ax.set(xlabel="Epoch", ylabel="MSE Loss", title=f"Diffusion Loss ({mode_tag})")
    ax.legend()
    fig.tight_layout()
    figs_out = figures_dir or FIGURES_DIR
    os.makedirs(figs_out, exist_ok=True)
    fig.savefig(os.path.join(figs_out, f"diff_loss_{mode_tag}.png"), dpi=150)
    plt.close(fig)

    return losses, list(zip(val_epochs_list, val_losses_list))


def _save_diff_ckpt(path, ae, unet, conditioner, latent_std, diff_losses, epoch,
                    diff_val_losses=None):
    torch.save({
        "ae":              ae.state_dict(),
        "unet":            unet.state_dict(),
        "conditioner":     conditioner.state_dict(),
        "latent_std":      latent_std.cpu(),
        "diff_losses":     diff_losses,
        "diff_val_losses": diff_val_losses or [],
        "epoch":           epoch,
    }, path)


# ── Synthesis / quick eval ────────────────────────────────────────────────────

@torch.no_grad()
def synthesize(ae, unet, conditioner, schedule, mri_vol, cond_data, latent_std,
               n_steps=200, device=DEVICE):
    cond = cond_data.squeeze(0) if cond_data.dim() == 2 else cond_data
    return synthesize_tau_pet(
        mri_vol, cond, ae, unet, schedule, conditioner.encode, latent_std,
        device=device, n_steps=n_steps,
    )


def quick_eval(ae, unet, conditioner, schedule, latent_std, test_ds,
               n_samples=10, mode="ptau217", device=DEVICE):
    maes, mses = [], []
    for i in range(min(n_samples, len(test_ds))):
        pet, mri, cond = test_ds[i]
        gen    = synthesize_tau_pet(
            mri.unsqueeze(0), cond, ae, unet, schedule,
            conditioner.encode, latent_std, device=device, n_steps=50,
        )
        pet_np = pet.squeeze().numpy()
        gen_np = gen.squeeze().numpy()
        maes.append(np.abs(pet_np - gen_np).mean())
        mses.append(((pet_np - gen_np) ** 2).mean())
    print(f"Eval ({mode}, n={len(maes)}):  MAE={np.mean(maes):.4f}  MSE={np.mean(mses):.6f}")
    return np.mean(maes), np.mean(mses)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train TauGenNet with optional LP and/or Modality-Aware AE loss")
    p.add_argument("--mode",            choices=["ptau217", "ptau217_mlp", "atrophy", "combined"],
                   required=True)
    p.add_argument("--ae-epochs",       type=int,   default=AE_EPOCHS)
    p.add_argument("--diff-epochs",     type=int,   default=DIFF_EPOCHS)
    p.add_argument("--batch-size",      type=int,   default=BATCH_SIZE)
    p.add_argument("--checkpoint-dir",  type=str,   default=CHECKPOINT_DIR)
    p.add_argument("--ae-checkpoint",   type=str,   default=AE_CHECKPOINT_PATH)
    p.add_argument("--skip-ae",         action="store_true",
                   help="Load AE from --ae-checkpoint instead of retraining")
    p.add_argument("--eval-only",       action="store_true",
                   help="Skip training; load checkpoint and run quick eval")
    p.add_argument("--patience",        type=int,   default=50)
    p.add_argument("--val-every",       type=int,   default=1)
    p.add_argument("--unet-channels",   type=str,   default="256,512,768")
    p.add_argument("--n-transformer",   type=int,   default=3)
    p.add_argument("--monitor-every",   type=int,   default=25)
    p.add_argument("--split",           type=str,   default="64/16/20")
    p.add_argument("--monitor-dir",     type=str,   default=None)
    p.add_argument("--figures-dir",     type=str,   default=None)
    p.add_argument("--use-mask",        action=argparse.BooleanOptionalAction, default=True,
                   help="Apply DK atlas mask to data (default: on; use --no-use-mask to disable)")
    p.add_argument("--fold",            type=int,   default=None)
    p.add_argument("--n-folds",         type=int,   default=5)
    p.add_argument("--lr",              type=float, default=LR)
    p.add_argument("--kl-weight",       type=float, default=1e-4,
                   help="KL weight in AE loss (default 1e-4)")
    p.add_argument("--latent-ch",       type=int,   default=LATENT_CH,
                   help=f"Latent channel count for AE (default {LATENT_CH})")
    # ── Auxiliary loss flags ──────────────────────────────────────────────────
    p.add_argument("--lp-weight",       type=float, default=0.0,
                   help="Weight for 3D Laplacian Pyramid loss. 0 = disabled (default).")
    p.add_argument("--modality-weight", type=float, default=0.0,
                   help="Weight for Modality-Aware loss. 0 = disabled (default).")
    p.add_argument("--modality-ckpt",   type=str,   default=None,
                   help="Path to modality discriminator checkpoint "
                        "(default: MaM-DiT epoch-60 checkpoint).")
    return p.parse_args()


def main():
    args   = parse_args()
    device = DEVICE
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    diff_ckpt = os.path.join(args.checkpoint_dir, f"diff_{args.mode}.pt")

    # ── Dataset ───────────────────────────────────────────────────────────────
    split_parts = [int(x) for x in args.split.split("/")]
    train_frac, val_frac = split_parts[0] / 100, split_parts[1] / 100

    fold_kwargs = {}
    if args.fold is not None:
        fold_kwargs = {"fold_idx": args.fold, "n_folds": args.n_folds}

    if args.mode == "combined":
        train_ds, val_ds, test_ds, train_loader, val_loader, _ = \
            _dataset_combined.build_dataloaders(
                mode=args.mode, batch_size=args.batch_size, use_dk_mask=args.use_mask,
                train_frac=train_frac, val_frac=val_frac, **fold_kwargs,
            )
    else:
        dataset_mode = "ptau217" if args.mode == "ptau217_mlp" else args.mode
        train_ds, val_ds, test_ds, train_loader, val_loader, _ = \
            _dataset_final.build_dataloaders(
                mode=dataset_mode, batch_size=args.batch_size, use_dk_mask=args.use_mask,
                train_frac=train_frac, val_frac=val_frac, **fold_kwargs,
            )
    print(f"Mode: {args.mode}  |  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ── Models ────────────────────────────────────────────────────────────────
    ae = Autoencoder3D(latent_ch=args.latent_ch).to(device)
    print(f"AE params: {sum(p.numel() for p in ae.parameters()):,}  latent_ch={args.latent_ch}")

    conditioner = None
    unet        = None
    schedule    = None
    if args.diff_epochs > 0:
        unet_ch     = tuple(int(x) for x in args.unet_channels.split(","))
        conditioner = build_conditioner(args.mode, device=device)
        unet        = DenoisingUNet3D(context_dim=conditioner.out_dim,
                                      ch_list=unet_ch,
                                      n_transformer=args.n_transformer).to(device)
        schedule    = DiffusionSchedule(device=device)
        print(f"UNet params: {sum(p.numel() for p in unet.parameters()):,}")

    # ── Auxiliary loss setup ──────────────────────────────────────────────────
    modality_loss_fn = None
    if args.modality_weight > 0.0:
        print(f"Loading ModalityLoss from: "
              f"{args.modality_ckpt or ModalityLoss.DEFAULT_CKPT}", flush=True)
        modality_loss_fn = ModalityLoss(ckpt_path=args.modality_ckpt).to(device)

    # ── Load or train AE ──────────────────────────────────────────────────────
    ae_losses = []
    if args.skip_ae or args.eval_only:
        if os.path.exists(args.ae_checkpoint):
            ckpt = torch.load(args.ae_checkpoint, map_location=device)
            ae.load_state_dict(ckpt["ae"])
            ae_losses = ckpt.get("ae_losses", [])
            print(f"Loaded AE from {args.ae_checkpoint}  ({len(ae_losses)} epochs)")
        else:
            print(f"Warning: AE checkpoint not found at {args.ae_checkpoint}")
    else:
        ae_losses = train_ae(
            ae, train_loader, args.ae_epochs, args.ae_checkpoint, device,
            val_loader=val_loader, patience=args.patience,
            val_every=1, figures_dir=args.figures_dir,
            kl_weight=args.kl_weight,
            lp_weight=args.lp_weight,
            modality_loss_fn=modality_loss_fn,
            modality_weight=args.modality_weight,
        )

    if args.diff_epochs == 0:
        print("--diff-epochs 0: skipping diffusion training.")
        return

    if args.eval_only:
        if os.path.exists(diff_ckpt):
            ckpt = torch.load(diff_ckpt, map_location=device)
            unet.load_state_dict(ckpt["unet"])
            if ckpt.get("conditioner"):
                conditioner.load_state_dict(ckpt["conditioner"])
            latent_std = ckpt.get("latent_std",
                                  torch.ones(1, LATENT_CH, 1, 1, 1)).to(device)
            print(f"Loaded diffusion model from {diff_ckpt}")
        else:
            latent_std = torch.ones(1, LATENT_CH, 1, 1, 1, device=device)
        quick_eval(ae, unet, conditioner, schedule, latent_std,
                   test_ds, mode=args.mode, device=device)
        return

    # ── Load or resume diffusion ──────────────────────────────────────────────
    start_epoch     = 0
    diff_losses     = []
    diff_val_losses = []
    latent_std      = None

    if os.path.exists(diff_ckpt):
        ckpt = torch.load(diff_ckpt, map_location=device)
        ae.load_state_dict(ckpt["ae"])
        unet.load_state_dict(ckpt["unet"])
        if ckpt.get("conditioner"):
            conditioner.load_state_dict(ckpt["conditioner"])
        latent_std = ckpt.get("latent_std", None)
        if latent_std is not None:
            latent_std = latent_std.to(device)
            print(f"Resumed latent_std: {latent_std.squeeze().tolist()}")
        diff_losses     = ckpt.get("diff_losses", [])
        diff_val_losses = ckpt.get("diff_val_losses", [])
        start_epoch     = ckpt.get("epoch", 0)
        print(f"Resuming diffusion from epoch {start_epoch}")

    # ── Pre-cache latents ─────────────────────────────────────────────────────
    print("Pre-computing latents (train)...")
    z_pets_tr, z_mris_tr, conds_tr = precompute_latents(ae, train_ds, device, "Caching train")

    print("Pre-computing latents (val)...")
    z_pets_val, z_mris_val, conds_val = precompute_latents(ae, val_ds, device, "Caching val")

    if latent_std is None:
        z_all      = torch.cat([z_pets_tr, z_mris_tr], dim=0)
        latent_std = z_all.std(dim=(0, 2, 3, 4), keepdim=True).to(device)
        print(f"latent_std per channel: {latent_std.squeeze().tolist()}")

    ls_cpu      = latent_std.cpu()
    z_pets_tr  /= ls_cpu;  z_mris_tr  /= ls_cpu
    z_pets_val /= ls_cpu;  z_mris_val /= ls_cpu

    lat_train_ds  = LatentDataset(z_pets_tr,  z_mris_tr,  conds_tr)
    lat_val_ds    = LatentDataset(z_pets_val, z_mris_val, conds_val)
    lat_train_ldr = DataLoader(lat_train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=0)
    lat_val_ldr   = DataLoader(lat_val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=0)

    # ── Train diffusion ───────────────────────────────────────────────────────
    new_losses, new_val_losses = train_diffusion(
        ae, unet, conditioner, schedule, latent_std,
        lat_train_ldr, lat_val_ldr, val_ds,
        args.diff_epochs, start_epoch, diff_ckpt, device,
        patience=args.patience,
        val_every=args.val_every,
        monitor_every=args.monitor_every,
        monitor_dir=args.monitor_dir,
        figures_dir=args.figures_dir,
        lr=args.lr,
    )
    diff_losses     += new_losses
    diff_val_losses += new_val_losses

    # ── Loss plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ae_losses)
    axes[0].set(title="AE Loss (LP+Modality)", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(diff_losses, label="Train", alpha=0.8)
    if diff_val_losses:
        ve, vl = zip(*diff_val_losses)
        axes[1].plot(ve, vl, "o-", ms=3, lw=1.2, label="Val")
    axes[1].legend()
    axes[1].set(title=f"Diffusion Loss ({args.mode})", xlabel="Epoch", ylabel="MSE")
    plt.tight_layout()
    out_fig = os.path.join(args.checkpoint_dir, f"losses_{args.mode}_lp_modality.png")
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Saved loss plot → {out_fig}")


if __name__ == "__main__":
    main()
