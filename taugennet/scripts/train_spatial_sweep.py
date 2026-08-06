#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
train_spatial_atrophy.py – Train TauGenNet with atlas-based spatial atrophy map (Option 2).

The atrophy map is a (1, 96, 112, 96) volume where each voxel's value equals the
patient's regional atrophy z-score for the DK region it belongs to, built from an
MNI152 DK atlas (data/raw/mni_dk_atlas.nii.gz).

At training time the map is avg-pooled to latent resolution (1, 12, 14, 12) and
concatenated with [z_t, z_m] as a 7th UNet input channel.
Cross-attention conditioning (atrophy_vec → 16 tokens) is kept in parallel.

Usage
-----
# Skip AE training (reuse existing AE checkpoint), train diffusion only:
python scripts/train_spatial_atrophy.py --skip-ae --ae-checkpoint results/checkpoints/taugennet_checkpoint_paper.pt

# Full run from scratch:
python scripts/train_spatial_atrophy.py

# Resume interrupted diffusion training:
python scripts/train_spatial_atrophy.py --skip-ae --ae-checkpoint results/checkpoints/taugennet_checkpoint_paper.pt

# Eval only:
python scripts/train_spatial_atrophy.py --eval-only
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
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.config import (DEVICE, T_STEPS, LR, AE_EPOCHS, DIFF_EPOCHS, BATCH_SIZE,
                        AE_CHECKPOINT_PATH, FIGURES_DIR, LATENT_CH, LATENT_SCALE)
from src import dataset_spatial as _dataset
from src.conditioning import CombinedConditioner, CombinedMLPConditioner
from src.models_spatial_sweep   import Autoencoder3D, DenoisingUNet3D
from src.diffusion         import DiffusionSchedule
from src.conditioning      import (AtrophyConditioner, NullConditioner, PTau217Conditioner,
                                   PTau217MLPConditioner)
from src.ema               import EMA

_DEFAULT_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "checkpoints", "spatial_atrophy")


# ── Loss ──────────────────────────────────────────────────────────────────────

def ae_loss(recon, x, mean, logvar, kl_weight=1e-4):
    recon_loss = F.l1_loss(recon, x)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss


# ── Latent cache ──────────────────────────────────────────────────────────────

class SpatialLatentDataset(Dataset):
    """Cached (z_pet, z_mri, atrophy_vec, atrophy_map_latent) tuples in CPU RAM."""
    def __init__(self, z_pets, z_mris, atrophy_vecs, atrophy_maps_latent):
        self.z_pets              = z_pets
        self.z_mris              = z_mris
        self.atrophy_vecs        = atrophy_vecs
        self.atrophy_maps_latent = atrophy_maps_latent

    def __len__(self):
        return len(self.z_pets)

    def __getitem__(self, idx):
        return (self.z_pets[idx], self.z_mris[idx],
                self.atrophy_vecs[idx], self.atrophy_maps_latent[idx])


def precompute_latents(ae, dataset, device, desc="Caching latents", use_tissue=False):
    """Encode volumes and downsample spatial-conditioning maps to latent resolution.

    The returned ``cond_latents`` carry the atrophy map (1 ch) and, when
    ``use_tissue``, the one-hot tissue fractions (3 ch) concatenated along the
    channel axis — so the diffusion loops can ``cat([zt, zm, cond_latent])``
    without caring how many conditioning channels there are.
    """
    ae.eval()
    z_pets, z_mris, atrophy_vecs, cond_latents = [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=desc, leave=False):
            if use_tissue:
                pet, mri, atrophy_vec, atrophy_map, tissue, _diagnosis = dataset[i]
            else:
                pet, mri, atrophy_vec, atrophy_map, _diagnosis = dataset[i]
            z_pets.append(ae.encode_mean(pet.unsqueeze(0).to(device)).cpu().squeeze(0))
            z_mris.append(ae.encode_mean(mri.unsqueeze(0).to(device)).cpu().squeeze(0))
            atrophy_vecs.append(atrophy_vec)
            # Downsample (1, 96, 112, 96) → (1, 12, 14, 12)
            cond = F.avg_pool3d(atrophy_map.unsqueeze(0), kernel_size=8).squeeze(0)
            if use_tissue:
                # one-hot (3, 96,112,96) → soft tissue fractions (3, 12, 14, 12)
                tissue_latent = F.avg_pool3d(tissue.unsqueeze(0), kernel_size=8).squeeze(0)
                cond = torch.cat([cond, tissue_latent], dim=0)   # (4, 12, 14, 12)
            cond_latents.append(cond)
    return (torch.stack(z_pets), torch.stack(z_mris),
            torch.stack(atrophy_vecs), torch.stack(cond_latents))


# ── AE pretraining ────────────────────────────────────────────────────────────

def train_ae(ae, train_loader, n_epochs, ckpt_path, device,
             val_loader=None, patience=20, val_every=5, figures_dir=None, kl_weight=1e-4):
    ae_opt = torch.optim.Adam(ae.parameters(), lr=LR)
    scaler = GradScaler('cuda')
    train_losses, val_epochs_list, val_losses_list = [], [], []
    best_val_loss  = float("inf")
    best_epoch     = 0
    epochs_no_improv = 0
    best_ckpt_path = ckpt_path.replace(".pt", "_best.pt")

    print("=== Autoencoder Pretraining ===", flush=True)
    start = time.time()
    for epoch in range(n_epochs):
        ae.train()
        epoch_loss = 0.0
        t0 = time.time()
        # dataset returns (pet, mri, atrophy_vec, atrophy_map, diagnosis) — ignore extras
        for pet, mri, *_ in train_loader:
            pet, mri = pet.to(device), mri.to(device)
            ae_opt.zero_grad()
            loss = torch.tensor(0.0, device=device)
            with autocast('cuda'):
                for vol in (pet, mri):
                    recon, mean, logvar = ae(vol)
                    loss = loss + ae_loss(recon, vol, mean, logvar, kl_weight=kl_weight)
            scaler.scale(loss).backward()
            scaler.step(ae_opt)
            scaler.update()
            epoch_loss += loss.item()

        avg = epoch_loss / len(train_loader)
        train_losses.append(avg)

        if val_loader is not None and (epoch + 1) % val_every == 0:
            ae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for pet, mri, *_ in val_loader:
                    pet, mri = pet.to(device), mri.to(device)
                    with autocast('cuda'):
                        for vol in (pet, mri):
                            recon, mean, logvar = ae(vol)
                            val_loss += ae_loss(recon, vol, mean, logvar, kl_weight=kl_weight).item()
            val_loss /= len(val_loader)
            val_epochs_list.append(epoch + 1)
            val_losses_list.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch + 1
                epochs_no_improv = 0
                torch.save({"ae": ae.state_dict(), "epoch": epoch + 1}, best_ckpt_path)
            else:
                epochs_no_improv += val_every
                if epochs_no_improv >= patience:
                    print(f"  Early stop at epoch {epoch+1} (best val={best_val_loss:.4f})")
                    break

        if (epoch + 1) % 5 == 0:
            elapsed   = time.time() - start
            remaining = (time.time() - t0) * (n_epochs - epoch - 1)
            val_str   = f"  val={val_losses_list[-1]:.4f}" if val_losses_list else ""
            print(f"  AE {epoch+1:3d}/{n_epochs}  loss={avg:.4f}{val_str}  "
                  f"elapsed={elapsed/60:.1f}m  remaining={remaining/60:.1f}m", flush=True)
            torch.save({"ae": ae.state_dict(), "ae_losses": train_losses, "epoch": epoch + 1},
                       ckpt_path)

    print(f"AE done. Total: {(time.time()-start)/60:.1f}m")
    return train_losses


# ── SSIM monitoring ───────────────────────────────────────────────────────────

@torch.no_grad()
def _monitor_ssim(ae, unet, conditioner, schedule, latent_std,
                  val_ds, epoch, monitor_dir, device, n_steps=50, use_tissue=False):
    """Quick SSIM on val_ds[0]. Runs a local DDPM loop that passes the
    spatial conditioning latent (atrophy map, + tissue one-hot if use_tissue)."""
    unet.eval(); ae.eval()
    if hasattr(conditioner, "eval"):
        conditioner.eval()

    if use_tissue:
        pet, mri, atrophy_vec, atrophy_map, tissue, _diag = val_ds[0]
    else:
        pet, mri, atrophy_vec, atrophy_map, _diag = val_ds[0]
    real_np = pet.squeeze().numpy()

    mri_b     = mri.unsqueeze(0).to(device)
    zm        = ae.encode_mean(mri_b) / latent_std
    am_latent = F.avg_pool3d(atrophy_map.unsqueeze(0).to(device), kernel_size=8)  # (1,1,12,14,12)
    if use_tissue:
        tissue_latent = F.avg_pool3d(tissue.unsqueeze(0).to(device), kernel_size=8)  # (1,3,...)
        am_latent     = torch.cat([am_latent, tissue_latent], dim=1)                 # (1,4,...)
    ctx       = conditioner.encode(atrophy_vec.unsqueeze(0).to(device))
    zt        = torch.randn_like(zm)

    step_indices = list(range(0, T_STEPS, T_STEPS // n_steps))[::-1]
    for t_idx in step_indices:
        t_batch = torch.full((1,), t_idx, device=device, dtype=torch.long)
        ht      = torch.cat([zt, zm, am_latent], dim=1)   # (1, 7, lH, lW, lD)
        eps_hat = unet(ht, t_batch, ctx, cond_map=am_latent)
        # DDPM reverse step (manual — p_sample hardcodes cat([zt, zm]) internally)
        z0_hat  = (zt - schedule.sqrt_one_m_ab[t_idx] * eps_hat) / schedule.sqrt_ab[t_idx]
        beta_t  = schedule.betas[t_idx]
        ab_prev = schedule.alpha_bar_prev[t_idx]
        ab_t    = schedule.alpha_bar[t_idx]
        coef1   = ab_prev.sqrt() * beta_t / (1 - ab_t)
        coef2   = schedule.alphas[t_idx].sqrt() * (1 - ab_prev) / (1 - ab_t)
        mean    = coef1 * z0_hat + coef2 * zt
        zt      = mean if t_idx == 0 else mean + schedule.post_var[t_idx].sqrt() * torch.randn_like(zt)

    gen    = ae.decode(zt * latent_std)
    gen_np = gen.squeeze().cpu().numpy()

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
    plt.savefig(os.path.join(monitor_dir, f"epoch_{epoch:04d}.png"), dpi=100)
    plt.close(fig)

    if hasattr(conditioner, "train"):
        conditioner.train()
    unet.train()
    return ssim_val


# ── Diffusion training ────────────────────────────────────────────────────────

def _save_ckpt(path, ae, unet, conditioner, latent_std, diff_losses, epoch,
               diff_val_losses=None, ema=None, noise_schedule="linear"):
    extra_cond_ch = unet.in_conv.in_channels - 2 * ae.latent_ch
    _cond_mode = {"AtrophyConditioner": "atrophy", "NullConditioner": "none",
                  "PTau217Conditioner": "ptau217",
                  "PTau217MLPConditioner": "ptau217_mlp"}.get(type(conditioner).__name__, "atrophy")
    ckpt = {
        "ae":             ae.state_dict(),
        "unet":           unet.state_dict(),   # raw weights (resume-safe)
        "conditioner":    conditioner.state_dict(),
        "latent_std":     latent_std.cpu(),
        "latent_ch":      ae.latent_ch,
        "extra_cond_ch":  extra_cond_ch,        # 1 = atrophy only; 4 = atrophy + tissue
        "use_tissue":     extra_cond_ch >= 4,
        "ae_scale":       getattr(ae, "scale", LATENT_SCALE),   # so inference rebuilds the exact AE
        "ae_res_blocks":  getattr(ae, "n_res_blocks", 1),
        "use_spade":      getattr(unet, "use_spade", False),  # multi-scale SPADE injection
        "cfg_prob":       getattr(unet, "cfg_prob", 0.0),      # >0 => CFG-trained, guidance-capable
        "cond_mode":      _cond_mode,           # atrophy | none | ptau217 (cross-attention)
        "noise_schedule": noise_schedule,       # linear | cosine — sampler MUST match this
        "diff_losses":    diff_losses,
        "diff_val_losses": diff_val_losses or [],
        "epoch":          epoch,
    }
    if ema is not None:
        ckpt["unet_ema"] = ema.shadow           # EMA weights (preferred at eval/inference)
    torch.save(ckpt, path)


def _compute_lr(epoch, base_lr, warmup_epochs, schedule, total_epochs):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(warmup_epochs, 1)
    if schedule == "cosine":
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 1e-6 + 0.5 * (base_lr - 1e-6) * (1 + math.cos(math.pi * progress))
    return base_lr


def train_diffusion(ae, unet, conditioner, schedule, latent_std,
                    lat_train_ldr, lat_val_ldr, val_ds,
                    n_epochs, start_epoch, ckpt_path, device,
                    patience=50, val_every=1, monitor_every=25,
                    monitor_dir=None, figures_dir=None, lr=LR, use_tissue=False,
                    ema_decay=0.0, min_snr_gamma=0.0, ema_state=None,
                    lr_warmup_epochs=0, lr_schedule="constant", cfg_prob=0.0):
    trainable = list(unet.parameters()) + conditioner.trainable_parameters()
    diff_opt  = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    ema = EMA(unet, decay=ema_decay) if ema_decay and ema_decay > 0 else None
    if ema is not None and ema_state is not None:
        ema.load_state_dict(ema_state); ema.to(device)
    if ema is not None:
        print(f"EMA enabled (decay={ema_decay})", flush=True)
    if min_snr_gamma and min_snr_gamma > 0:
        print(f"Min-SNR weighting enabled (gamma={min_snr_gamma})", flush=True)

    losses, val_epochs_list, val_losses_list = [], [], []
    best_val_loss    = float("inf")
    best_epoch       = start_epoch
    epochs_no_improv = 0
    best_ckpt_path   = ckpt_path.replace(".pt", "_best.pt")
    ae.eval()

    print(f"=== Spatial Atrophy Diffusion Training (start={start_epoch}, patience={patience}) ===",
          flush=True)
    start = time.time()

    for epoch in range(start_epoch, n_epochs):
        cur_lr = _compute_lr(epoch, lr, lr_warmup_epochs, lr_schedule, n_epochs)
        for pg in diff_opt.param_groups:
            pg["lr"] = cur_lr
        unet.train()
        if hasattr(conditioner, "train"):
            conditioner.train()
        epoch_loss = 0.0
        t0 = time.time()

        unet.cfg_prob = cfg_prob   # stash so _save_ckpt records it (guidance-capable flag)
        for z0, zm, atrophy_vec, am_latent in lat_train_ldr:
            z0, zm        = z0.to(device), zm.to(device)
            atrophy_vec   = atrophy_vec.to(device)
            am_latent     = am_latent.to(device)    # (B, 1, 12, 14, 12)

            B       = z0.shape[0]
            t       = torch.randint(0, T_STEPS, (B,), device=device)
            zt, eps = schedule.q_sample(z0, t)

            ctx     = conditioner.encode(atrophy_vec)
            # CFG dropout: for a random subset, zero BOTH the atrophy spatial map (channel concat)
            # and the cross-attention context -> the model also learns unconditional denoising, which
            # guided sampling needs. z_m (MRI) always stays (it's the structural backbone, not guided).
            if cfg_prob > 0:
                keep      = (torch.rand(B, device=device) >= cfg_prob).float()
                ctx       = ctx * keep.view(B, 1, 1)
                am_latent = am_latent * keep.view(B, 1, 1, 1, 1)
            ht      = torch.cat([zt, zm, am_latent], dim=1)   # (B, 7, 12, 14, 12)
            eps_hat = unet(ht, t, ctx, cond_map=am_latent)
            if min_snr_gamma and min_snr_gamma > 0:
                snr  = schedule.alpha_bar[t] / (1 - schedule.alpha_bar[t])   # (B,)
                w    = torch.clamp(snr, max=min_snr_gamma) / snr             # eps-pred form
                per  = ((eps_hat - eps) ** 2).mean(dim=[1, 2, 3, 4])
                loss = (w * per).mean()
            else:
                loss = F.mse_loss(eps_hat, eps)

            diff_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            diff_opt.step()
            if ema is not None:
                ema.update(unet)
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(lat_train_ldr)
        losses.append(avg_train)
        elapsed   = time.time() - start
        remaining = (time.time() - t0) * (n_epochs - epoch - 1)

        val_str = ""
        if (epoch + 1) % val_every == 0:
            unet.eval()
            if hasattr(conditioner, "eval"):
                conditioner.eval()
            if ema is not None:
                ema.store(unet); ema.copy_to(unet)   # validate the EMA weights
            val_loss = 0.0
            with torch.no_grad():
                for z0, zm, atrophy_vec, am_latent in lat_val_ldr:
                    z0, zm      = z0.to(device), zm.to(device)
                    atrophy_vec = atrophy_vec.to(device)
                    am_latent   = am_latent.to(device)
                    B           = z0.shape[0]
                    t           = torch.randint(0, T_STEPS, (B,), device=device)
                    zt, eps     = schedule.q_sample(z0, t)
                    ctx         = conditioner.encode(atrophy_vec)
                    ht          = torch.cat([zt, zm, am_latent], dim=1)
                    eps_hat     = unet(ht, t, ctx, cond_map=am_latent)
                    val_loss   += F.mse_loss(eps_hat, eps).item()
            val_loss /= len(lat_val_ldr)
            if ema is not None:
                ema.restore(unet)                    # back to raw weights for training/saving
            val_epochs_list.append(epoch + 1)
            val_losses_list.append(val_loss)
            val_str = f"  val={val_loss:.6f}"

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_epoch       = epoch + 1
                epochs_no_improv = 0
                _save_ckpt(best_ckpt_path, ae, unet, conditioner, latent_std,
                           losses, epoch + 1,
                           diff_val_losses=list(zip(val_epochs_list, val_losses_list)),
                           ema=ema, noise_schedule=schedule.schedule_name)
            else:
                epochs_no_improv += val_every

        ssim_str = ""
        if (epoch + 1) % monitor_every == 0:
            if ema is not None:
                ema.store(unet); ema.copy_to(unet)
            ssim_val = _monitor_ssim(ae, unet, conditioner, schedule, latent_std,
                                     val_ds, epoch + 1,
                                     monitor_dir or os.path.join(FIGURES_DIR, "spatial_atrophy", "monitor"),
                                     device, use_tissue=use_tissue)
            if ema is not None:
                ema.restore(unet)
            ssim_str = f"  SSIM={ssim_val:.3f}"

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}"
                  f"  train={avg_train:.6f}{val_str}{ssim_str}"
                  f"  elapsed={elapsed/60:.1f}m  remaining={remaining/60:.1f}m", flush=True)
            _save_ckpt(ckpt_path, ae, unet, conditioner, latent_std,
                       losses, epoch + 1,
                       diff_val_losses=list(zip(val_epochs_list, val_losses_list)),
                       ema=ema, noise_schedule=schedule.schedule_name)

        if epochs_no_improv >= patience:
            print(f"Early stopping at epoch {epoch+1} (best val={best_val_loss:.6f})")
            break

    print(f"Diffusion done. Total: {(time.time()-start)/60:.1f}m")
    return losses, list(zip(val_epochs_list, val_losses_list))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train TauGenNet — spatial atrophy option")
    p.add_argument("--ae-epochs",      type=int,   default=AE_EPOCHS)
    p.add_argument("--diff-epochs",    type=int,   default=DIFF_EPOCHS)
    p.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    p.add_argument("--checkpoint-dir", type=str,   default=_DEFAULT_CKPT_DIR)
    p.add_argument("--ae-checkpoint",  type=str,   default=AE_CHECKPOINT_PATH)
    p.add_argument("--skip-ae",        action="store_true")
    p.add_argument("--eval-only",      action="store_true")
    p.add_argument("--patience",       type=int,   default=50)
    p.add_argument("--val-every",      type=int,   default=1)
    p.add_argument("--unet-channels",  type=str,   default="256,512,768")
    p.add_argument("--n-transformer",  type=int,   default=3)
    p.add_argument("--monitor-every",  type=int,   default=25)
    p.add_argument("--split",          type=str,   default="64/16/20")
    p.add_argument("--monitor-dir",    type=str,   default=None)
    p.add_argument("--figures-dir",    type=str,   default=None)
    p.add_argument("--use-mask",       action=argparse.BooleanOptionalAction, default=True,
                   help="Apply DK atlas mask to data (default: on; use --no-use-mask to disable)")
    p.add_argument("--use-mentor-split", action=argparse.BooleanOptionalAction, default=True,
                   help="Use Anil's mentor split (default). --no-use-mentor-split forces the "
                        "legacy random fold split — required to stay comparable to legacy-trained "
                        "models (e.g. the cond_a/cond_b ablation set).")
    p.add_argument("--atlas-path",     type=str,   default=None,
                   help="Path to MNI DK atlas NIfTI (default: data/raw/mni_dk_atlas.nii.gz)")
    p.add_argument("--fold",           type=int,   default=None)
    p.add_argument("--n-folds",        type=int,   default=5)
    p.add_argument("--no-val-split",   action="store_true",
                   help="Fold val into train (train on ALL dev); pair with huge --val-every/--patience.")
    p.add_argument("--latent-scale",   type=int,   default=LATENT_SCALE, choices=[4, 8, 16],
                   help="AE downsampling factor = latent spatial resolution (8 -> 12x14x12).")
    p.add_argument("--ae-res-blocks",  type=int,   default=2,
                   help="AE residual blocks per encoder/decoder level. Default 2 = paper spec. "
                        "Reusing an AE (--skip-ae) always uses that checkpoint's own value.")
    p.add_argument("--latent-ch",      type=int,   default=LATENT_CH,
                   help="Latent channel count (must match the AE checkpoint)")
    p.add_argument("--use-spade",      action="store_true",
                   help="Multi-scale SPADE conditioning injection: modulate every UNet stage with "
                        "the atrophy(+tissue) map, not just a single input concat. Targets WHERE "
                        "tau burden goes (spatial localization). New arch -> trains from scratch.")
    p.add_argument("--cfg-prob",       type=float, default=0.0,
                   help="Classifier-free guidance dropout prob. >0 zeros the atrophy map + "
                        "cross-attention context for that fraction of samples during training, so "
                        "the model also learns unconditional denoising. Required to use "
                        "--guidance-scale>1 at inference (generate_spatial.py). 0 = no CFG.")
    p.add_argument("--use-tissue",     action="store_true",
                   help="Add GM/WM/CSF tissue one-hot (T1_seg_in_MNI) as 3 extra "
                        "spatial conditioning channels (UNet in_ch 7 → 10)")
    p.add_argument("--phase6-manifest", type=str, default=None,
                   help="Pooled AD+MCI+CN manifest CSV; routes through dataset_phase6.")
    p.add_argument("--cond-mode",      type=str, default="atrophy",
                   choices=["atrophy", "none", "ptau217", "ptau217_mlp"],
                   help="Cross-attention conditioner alongside the spatial atrophy map: "
                        "'atrophy' = atrophy MLP (default); 'none' = learned-null context "
                        "(spatial map only); 'ptau217' = frozen CLIP on plasma p-tau217.")
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--kl-weight",      type=float, default=1e-4)
    p.add_argument("--ema-decay",      type=float, default=0.0,
                   help="EMA decay for the diffusion UNet; 0 = OFF (default). Typical: 0.999.")
    p.add_argument("--min-snr-gamma",  type=float, default=0.0,
                   help="Min-SNR-gamma loss weighting (eps-pred); 0 = OFF (default). Typical: 5.")
    p.add_argument("--lr-warmup-epochs", type=int, default=0,
                   help="Linear LR warmup for this many epochs before the main schedule.")
    p.add_argument("--lr-schedule",    choices=["constant", "cosine"], default="constant",
                   help="LR schedule after warmup: constant (default) or cosine decay to 1e-6.")
    p.add_argument("--noise-schedule", choices=["linear", "cosine"], default="linear",
                   help="Diffusion beta schedule: linear (default) or cosine (Nichol 2021).")
    return p.parse_args()


def main():
    args   = parse_args()
    device = DEVICE
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    diff_ckpt = os.path.join(args.checkpoint_dir, "diff_spatial_atrophy.pt")

    split_parts = [int(x) for x in args.split.split("/")]
    train_frac, val_frac = split_parts[0] / 100, split_parts[1] / 100
    fold_kwargs = {} if args.fold is None else {"fold_idx": args.fold, "n_folds": args.n_folds}

    # ptau217_mlp uses the SAME data as ptau217 (the scalar plasma value); only the cross-attention
    # conditioner differs (learned MLP vs frozen CLIP). Map it for the dataset.
    ds_cond_mode = "ptau217" if args.cond_mode == "ptau217_mlp" else args.cond_mode
    if getattr(args, "phase6_manifest", None):
        from src import dataset_phase6 as _p6
        _ds_cm = "combined" if args.cond_mode in ("combined", "combined_mlp") else ds_cond_mode
        train_ds, val_ds, test_ds, train_loader, val_loader, _ = _p6.build_dataloaders(
            manifest=args.phase6_manifest, batch_size=args.batch_size, use_dk_mask=args.use_mask,
            atlas_path=args.atlas_path, use_tissue=args.use_tissue, cond_mode=_ds_cm,
            no_val_split=args.no_val_split, **fold_kwargs,
        )
    else:
        train_ds, val_ds, test_ds, train_loader, val_loader, _ = _dataset.build_dataloaders(
            batch_size=args.batch_size, use_dk_mask=args.use_mask,
            train_frac=train_frac, val_frac=val_frac,
            atlas_path=args.atlas_path, use_tissue=args.use_tissue,
            cond_mode=ds_cond_mode, use_mentor_split=args.use_mentor_split,
            no_val_split=args.no_val_split, **fold_kwargs,
        )
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # Build the AE at the checkpoint's architecture when reusing one (--skip-ae); otherwise at the
    # CLI spec. Pre-2026-07 checkpoints have no ae_scale/ae_res_blocks -> legacy 8x / 1 block.
    ae_scale, ae_rb = args.latent_scale, args.ae_res_blocks
    if (args.skip_ae or args.eval_only) and os.path.exists(args.ae_checkpoint):
        _peek    = torch.load(args.ae_checkpoint, map_location="cpu")
        ae_scale = _peek.get("ae_scale", LATENT_SCALE)
        ae_rb    = _peek.get("ae_res_blocks", 1)
        del _peek
        print(f"AE arch taken from checkpoint: scale={ae_scale}  res_blocks/level={ae_rb}")
    ae = Autoencoder3D(latent_ch=args.latent_ch, scale=ae_scale, n_res_blocks=ae_rb).to(device)
    print(f"AE: latent_ch={args.latent_ch}  scale={ae_scale}  res_blocks/level={ae_rb}")
    print(f"AE params: {sum(p.numel() for p in ae.parameters()):,}")

    conditioner = None
    unet        = None
    schedule    = None
    if args.diff_epochs > 0 or args.eval_only:
        unet_ch       = tuple(int(x) for x in args.unet_channels.split(","))
        extra_cond_ch = 4 if args.use_tissue else 1   # 1 atrophy (+3 tissue one-hot)
        # Cross-attention conditioner: atrophy MLP / learned-null / frozen-CLIP ptau.
        # The spatial atrophy MAP is always present (extra_cond_ch); cond-mode only
        # changes what (if anything) drives cross-attention.
        if args.cond_mode == "atrophy":
            conditioner = AtrophyConditioner().to(device)
        elif args.cond_mode == "none":
            conditioner = NullConditioner().to(device)
        elif args.cond_mode == "ptau217":
            conditioner = PTau217Conditioner(device=device)
        elif args.cond_mode == "ptau217_mlp":
            conditioner = PTau217MLPConditioner().to(device)   # MLP on the scalar ptau (vs frozen CLIP)
        elif args.cond_mode == "combined":
            conditioner = CombinedConditioner(device=device)
        elif args.cond_mode == "combined_mlp":
            conditioner = CombinedMLPConditioner(device=device)
        else:
            raise ValueError(f"Unknown --cond-mode {args.cond_mode!r}")
        print(f"cond-mode: {args.cond_mode}  ({type(conditioner).__name__})")
        unet          = DenoisingUNet3D(latent_ch=args.latent_ch,
                                        context_dim=conditioner.out_dim,
                                        ch_list=unet_ch,
                                        n_transformer=args.n_transformer,
                                        extra_cond_ch=extra_cond_ch,
                                        use_spade=args.use_spade).to(device)
        if args.use_spade:
            print("SPADE multi-scale conditioning injection: ON")
        schedule      = DiffusionSchedule(device=device, schedule=args.noise_schedule)
        in_ch = args.latent_ch * 2 + extra_cond_ch
        print(f"UNet params (spatial, {in_ch}ch): {sum(p.numel() for p in unet.parameters()):,}")
        print(f"Conditioner params: {sum(p.numel() for p in conditioner.trainable_parameters()):,}")

    # ── Load or train AE ──────────────────────────────────────────────────────
    if args.skip_ae or args.eval_only:
        if os.path.exists(args.ae_checkpoint):
            ckpt = torch.load(args.ae_checkpoint, map_location=device)
            ae.load_state_dict(ckpt["ae"])
            print(f"Loaded AE from {args.ae_checkpoint}")
        else:
            print(f"Warning: AE checkpoint not found at {args.ae_checkpoint}")
    else:
        train_ae(ae, train_loader, args.ae_epochs, args.ae_checkpoint, device,
                 val_loader=val_loader, patience=args.patience,
                 val_every=5, figures_dir=args.figures_dir, kl_weight=args.kl_weight)

    if args.diff_epochs == 0:
        print("--diff-epochs 0: skipping diffusion training.")
        return

    # ── Load or resume diffusion ──────────────────────────────────────────────
    start_epoch = 0
    latent_std  = None
    ema_state   = None
    if os.path.exists(diff_ckpt):
        ckpt = torch.load(diff_ckpt, map_location=device)
        ae.load_state_dict(ckpt["ae"])
        unet.load_state_dict(ckpt["unet"])
        ema_state = ckpt.get("unet_ema", None)   # resume EMA shadow if present
        if ckpt.get("conditioner"):
            conditioner.load_state_dict(ckpt["conditioner"])
        latent_std  = ckpt.get("latent_std")
        if latent_std is not None:
            latent_std = latent_std.to(device)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resuming from epoch {start_epoch}")

    if args.eval_only:
        print("eval-only: quick MAE on test set")
        # basic evaluation stub — run full evaluate_final.py for complete metrics
        return

    # ── Pre-cache latents ─────────────────────────────────────────────────────
    print("Pre-computing latents (train)…")
    z_pets_tr, z_mris_tr, atrophy_vecs_tr, am_latents_tr = precompute_latents(
        ae, train_ds, device, "Caching train", use_tissue=args.use_tissue
    )
    print("Pre-computing latents (val)…")
    if len(val_ds) > 0:
        z_pets_val, z_mris_val, atrophy_vecs_val, am_latents_val = precompute_latents(
            ae, val_ds, device, "Caching val", use_tissue=args.use_tissue
        )
    else:
        # --no-val-split leaves val empty, and torch.stack([]) raises. Zero-length
        # tensors with the train shapes keep every downstream consumer working
        # unchanged; the val loop never runs because --val-every exceeds --diff-epochs.
        print("  (val empty -- skipping val latents)")
        z_pets_val       = z_pets_tr[:0].clone()
        z_mris_val       = z_mris_tr[:0].clone()
        atrophy_vecs_val = atrophy_vecs_tr[:0].clone()
        am_latents_val   = am_latents_tr[:0].clone()

    if latent_std is None:
        z_all      = torch.cat([z_pets_tr, z_mris_tr], dim=0)
        latent_std = z_all.std(dim=(0, 2, 3, 4), keepdim=True).to(device)
        print(f"latent_std: {latent_std.squeeze().tolist()}")

    ls_cpu = latent_std.cpu()
    z_pets_tr  /= ls_cpu;  z_mris_tr  /= ls_cpu
    z_pets_val /= ls_cpu;  z_mris_val /= ls_cpu

    lat_train_ds  = SpatialLatentDataset(z_pets_tr,  z_mris_tr,  atrophy_vecs_tr,  am_latents_tr)
    lat_val_ds    = SpatialLatentDataset(z_pets_val, z_mris_val, atrophy_vecs_val, am_latents_val)
    lat_train_ldr = DataLoader(lat_train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=0)
    lat_val_ldr   = DataLoader(lat_val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=0)

    # ── Train ─────────────────────────────────────────────────────────────────
    figs_dir = args.figures_dir or os.path.join(FIGURES_DIR, "spatial_atrophy")
    train_diffusion(
        ae, unet, conditioner, schedule, latent_std,
        lat_train_ldr, lat_val_ldr, val_ds,
        args.diff_epochs, start_epoch, diff_ckpt, device,
        patience=args.patience,
        val_every=args.val_every,
        monitor_every=args.monitor_every,
        monitor_dir=args.monitor_dir,
        figures_dir=figs_dir,
        lr=args.lr,
        use_tissue=args.use_tissue,
        ema_decay=args.ema_decay,
        min_snr_gamma=args.min_snr_gamma,
        ema_state=ema_state,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_schedule=args.lr_schedule,
        cfg_prob=args.cfg_prob,
    )


if __name__ == "__main__":
    main()
