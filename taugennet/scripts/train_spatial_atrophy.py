#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
train_spatial_atrophy.py – Train TauGenNet with atlas-based spatial atrophy map (Option 2).

The atrophy map is a (1, 96, 112, 96) volume where each voxel's value equals the
patient's regional atrophy z-score for the DK region it belongs to, built from an
MNI152 DK atlas (data/raw/mni_dk_atlas.nii.gz).

At training time the map is downsampled to latent resolution (1, 12, 14, 12) — via
avg_pool3d by default, or a learned AtrophyEncoder3D CNN with --use-atrophy-encoder
(spatial-injection Design B) — and concatenated with [z_t, z_m] as a 7th UNet input
channel. Cross-attention conditioning (atrophy_vec → 16 tokens) is kept in parallel.

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
                        AE_CHECKPOINT_PATH, FIGURES_DIR, LATENT_CH)
from src import dataset_spatial as _dataset
from src.models_spatial   import Autoencoder3D, DenoisingUNet3D
from src.diffusion         import DiffusionSchedule
from src.conditioning      import AtrophyConditioner, NullConditioner, PTau217Conditioner
from src.ema               import EMA

_DEFAULT_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "checkpoints", "spatial_atrophy")


# ── Loss ──────────────────────────────────────────────────────────────────────

def ae_loss(recon, x, mean, logvar, kl_weight=1e-4):
    recon_loss = F.l1_loss(recon, x)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss


# ── Latent cache ──────────────────────────────────────────────────────────────

class SpatialLatentDataset(Dataset):
    """Cached (z_pet, z_mri, atrophy_vec, atrophy_map, [tissue]) tuples in CPU RAM.

    atrophy_map (and tissue, if present) are cached at FULL resolution (1/3, 96,112,96),
    not pre-pooled — pooling (or the learned AtrophyEncoder3D) now happens inside
    DenoisingUNet3D.forward, since a learned encoder needs gradients on every forward
    and can't be precomputed once like the frozen-AE latents.
    """
    def __init__(self, z_pets, z_mris, atrophy_vecs, atrophy_maps, tissue_maps=None):
        self.z_pets       = z_pets
        self.z_mris       = z_mris
        self.atrophy_vecs = atrophy_vecs
        self.atrophy_maps = atrophy_maps
        self.tissue_maps  = tissue_maps

    def __len__(self):
        return len(self.z_pets)

    def __getitem__(self, idx):
        if self.tissue_maps is not None:
            return (self.z_pets[idx], self.z_mris[idx], self.atrophy_vecs[idx],
                    self.atrophy_maps[idx], self.tissue_maps[idx])
        return (self.z_pets[idx], self.z_mris[idx],
                self.atrophy_vecs[idx], self.atrophy_maps[idx])


def precompute_latents(ae, dataset, device, desc="Caching latents", use_tissue=False):
    """Encode PET/MRI to AE latents (frozen AE, safe to cache); keep the atrophy map
    (and tissue one-hot, if used) at full resolution — no pooling here, since that now
    happens inside DenoisingUNet3D.forward (needed for the learned-encoder path).
    """
    ae.eval()
    z_pets, z_mris, atrophy_vecs, atrophy_maps, tissue_maps = [], [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=desc, leave=False):
            if use_tissue:
                pet, mri, atrophy_vec, atrophy_map, tissue, _diagnosis = dataset[i]
                tissue_maps.append(tissue)
            else:
                pet, mri, atrophy_vec, atrophy_map, _diagnosis = dataset[i]
            z_pets.append(ae.encode_mean(pet.unsqueeze(0).to(device)).cpu().squeeze(0))
            z_mris.append(ae.encode_mean(mri.unsqueeze(0).to(device)).cpu().squeeze(0))
            atrophy_vecs.append(atrophy_vec)
            atrophy_maps.append(atrophy_map)
    return (torch.stack(z_pets), torch.stack(z_mris),
            torch.stack(atrophy_vecs), torch.stack(atrophy_maps),
            torch.stack(tissue_maps) if use_tissue else None)


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

    mri_b       = mri.unsqueeze(0).to(device)
    zm          = ae.encode_mean(mri_b) / latent_std
    atrophy_map_b = atrophy_map.unsqueeze(0).to(device)   # (1,1,96,112,96)
    tissue_b    = tissue.unsqueeze(0).to(device) if use_tissue else None
    ctx       = conditioner.encode(atrophy_vec.unsqueeze(0).to(device))
    zt        = torch.randn_like(zm)

    step_indices = list(range(0, T_STEPS, T_STEPS // n_steps))[::-1]
    for t_idx in step_indices:
        t_batch = torch.full((1,), t_idx, device=device, dtype=torch.long)
        eps_hat = unet(zt, zm, t_batch, ctx, atrophy_map_b, tissue_map=tissue_b)
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
                  "PTau217Conditioner": "ptau217"}.get(type(conditioner).__name__, "atrophy")
    ckpt = {
        "ae":             ae.state_dict(),
        "unet":           unet.state_dict(),   # raw weights (resume-safe); covers
                                                # atrophy_encoder/SPADE params too (submodules)
        "conditioner":    conditioner.state_dict(),
        "latent_std":     latent_std.cpu(),
        "latent_ch":      ae.latent_ch,
        "ae_res_blocks":  getattr(ae, "n_res_blocks", 1),  # must match at generate/eval time
        "extra_cond_ch":  extra_cond_ch,        # 1 = atrophy only; 4 = atrophy + tissue
        "use_tissue":     extra_cond_ch >= 4,
        "cond_mode":      _cond_mode,           # atrophy | none | ptau217 (cross-attention)
        "noise_schedule": noise_schedule,       # linear | cosine — sampler MUST match this
        "use_atrophy_encoder": getattr(unet, "use_atrophy_encoder", False),
        "atrophy_encoder_width": (
            unet.atrophy_encoder.net[0].out_channels
            if getattr(unet, "atrophy_encoder", None) is not None else None
        ),
        "use_spade":      getattr(unet, "use_spade", False),
        "spade_hidden": (
            unet.enc1.norm1.shared[0].out_channels
            if getattr(unet, "use_spade", False) else None
        ),
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


def _split_batch(batch, use_tissue):
    """lat_*_ldr batches are (z0, zm, atrophy_vec, atrophy_map[, tissue])."""
    if use_tissue:
        z0, zm, atrophy_vec, atrophy_map, tissue = batch
        return z0, zm, atrophy_vec, atrophy_map, tissue
    z0, zm, atrophy_vec, atrophy_map = batch
    return z0, zm, atrophy_vec, atrophy_map, None


def train_diffusion(ae, unet, conditioner, schedule, latent_std,
                    lat_train_ldr, lat_val_ldr, val_ds,
                    n_epochs, start_epoch, ckpt_path, device,
                    patience=50, val_every=1, monitor_every=25,
                    monitor_dir=None, figures_dir=None, lr=LR, use_tissue=False,
                    ema_decay=0.0, min_snr_gamma=0.0, ema_state=None,
                    lr_warmup_epochs=0, lr_schedule="constant"):
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

        for batch in lat_train_ldr:
            z0, zm, atrophy_vec, atrophy_map, tissue = _split_batch(batch, use_tissue)
            z0, zm        = z0.to(device), zm.to(device)
            atrophy_vec   = atrophy_vec.to(device)
            atrophy_map   = atrophy_map.to(device)   # (B, 1, 96, 112, 96)
            tissue        = tissue.to(device) if tissue is not None else None

            B       = z0.shape[0]
            t       = torch.randint(0, T_STEPS, (B,), device=device)
            zt, eps = schedule.q_sample(z0, t)

            ctx     = conditioner.encode(atrophy_vec)
            eps_hat = unet(zt, zm, t, ctx, atrophy_map, tissue_map=tissue)
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
                for batch in lat_val_ldr:
                    z0, zm, atrophy_vec, atrophy_map, tissue = _split_batch(batch, use_tissue)
                    z0, zm      = z0.to(device), zm.to(device)
                    atrophy_vec = atrophy_vec.to(device)
                    atrophy_map = atrophy_map.to(device)
                    tissue      = tissue.to(device) if tissue is not None else None
                    B           = z0.shape[0]
                    t           = torch.randint(0, T_STEPS, (B,), device=device)
                    zt, eps     = schedule.q_sample(z0, t)
                    ctx         = conditioner.encode(atrophy_vec)
                    eps_hat     = unet(zt, zm, t, ctx, atrophy_map, tissue_map=tissue)
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
                   help="Fold the validation set into train (train on ALL dev). For the final "
                        "fixed-epoch model; pair with huge --val-every/--patience to skip validation.")
    p.add_argument("--latent-ch",      type=int,   default=LATENT_CH,
                   help="Latent channel count (must match the AE checkpoint)")
    p.add_argument("--ae-res-blocks",  type=int,   default=1,
                   help="ResBlock3D count per AE encoder/decoder level (must match the AE "
                        "checkpoint's architecture, not just its latent_ch). Default 1 "
                        "(original layout); the canonical paper AE uses 2.")
    p.add_argument("--use-tissue",     action="store_true",
                   help="Add GM/WM/CSF tissue one-hot (T1_seg_in_MNI) as 3 extra "
                        "spatial conditioning channels (UNet in_ch 7 → 10)")
    p.add_argument("--use-atrophy-encoder", action="store_true",
                   help="Replace the raw avg_pool3d atrophy-map downsample with a learned "
                        "AtrophyEncoder3D CNN (spatial-injection Design B). Default OFF "
                        "(matches the cfgm_5 avg-pool baseline, SET3=0.510). NOTE: resuming "
                        "a run must pass the same flag it was launched with — this is not "
                        "read back from the checkpoint before construction.")
    p.add_argument("--atrophy-encoder-width", type=int, default=32,
                   help="Hidden channel width for AtrophyEncoder3D (only used with "
                        "--use-atrophy-encoder). Default 32 (matches GROUPNORM_GROUPS).")
    p.add_argument("--use-spade",      action="store_true",
                   help="Inject the RAW atrophy map at every UNet level via a SPADE "
                        "layer (spatial-injection Design A), in addition to the single "
                        "input-concat injection. Default OFF. Orthogonal to "
                        "--use-atrophy-encoder (both may be set together).")
    p.add_argument("--spade-hidden",   type=int, default=64,
                   help="Hidden channel width for each SPADE layer's shared conv "
                        "(only used with --use-spade).")
    p.add_argument("--cond-mode",      type=str, default="atrophy",
                   choices=["atrophy", "none", "ptau217"],
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

    train_ds, val_ds, test_ds, train_loader, val_loader, _ = _dataset.build_dataloaders(
        batch_size=args.batch_size, use_dk_mask=args.use_mask,
        train_frac=train_frac, val_frac=val_frac,
        atlas_path=args.atlas_path, use_tissue=args.use_tissue,
        cond_mode=args.cond_mode, use_mentor_split=args.use_mentor_split,
        no_val_split=args.no_val_split, **fold_kwargs,
    )
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    ae = Autoencoder3D(latent_ch=args.latent_ch, n_res_blocks=args.ae_res_blocks).to(device)
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
        else:
            raise ValueError(f"Unknown --cond-mode {args.cond_mode!r}")
        print(f"cond-mode: {args.cond_mode}  ({type(conditioner).__name__})")
        unet          = DenoisingUNet3D(latent_ch=args.latent_ch,
                                        context_dim=conditioner.out_dim,
                                        ch_list=unet_ch,
                                        n_transformer=args.n_transformer,
                                        extra_cond_ch=extra_cond_ch,
                                        use_atrophy_encoder=args.use_atrophy_encoder,
                                        atrophy_encoder_width=args.atrophy_encoder_width,
                                        use_spade=args.use_spade,
                                        spade_hidden=args.spade_hidden).to(device)
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
    z_pets_tr, z_mris_tr, atrophy_vecs_tr, atrophy_maps_tr, tissue_maps_tr = precompute_latents(
        ae, train_ds, device, "Caching train", use_tissue=args.use_tissue
    )
    print("Pre-computing latents (val)…")
    z_pets_val, z_mris_val, atrophy_vecs_val, atrophy_maps_val, tissue_maps_val = precompute_latents(
        ae, val_ds, device, "Caching val", use_tissue=args.use_tissue
    )

    if latent_std is None:
        z_all      = torch.cat([z_pets_tr, z_mris_tr], dim=0)
        latent_std = z_all.std(dim=(0, 2, 3, 4), keepdim=True).to(device)
        print(f"latent_std: {latent_std.squeeze().tolist()}")

    ls_cpu = latent_std.cpu()
    z_pets_tr  /= ls_cpu;  z_mris_tr  /= ls_cpu
    z_pets_val /= ls_cpu;  z_mris_val /= ls_cpu

    lat_train_ds  = SpatialLatentDataset(z_pets_tr,  z_mris_tr,  atrophy_vecs_tr,  atrophy_maps_tr,
                                         tissue_maps=tissue_maps_tr)
    lat_val_ds    = SpatialLatentDataset(z_pets_val, z_mris_val, atrophy_vecs_val, atrophy_maps_val,
                                         tissue_maps=tissue_maps_val)
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
    )


if __name__ == "__main__":
    main()
