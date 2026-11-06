#!/usr/bin/env python3
"""
train.py – train TauGenNet with either conditioning mode.

Usage
-----
# AE only (shared across both diffusion modes)
python train.py --mode atrophy --diff-epochs 0

# Diffusion training (AE loaded from checkpoint)
python train.py --mode atrophy --skip-ae --diff-epochs 2000

# Resume interrupted diffusion training
python train.py --mode atrophy --skip-ae

# Evaluate after training
python train.py --mode atrophy --eval-only
"""

import argparse
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
                        AE_CHECKPOINT_PATH, CHECKPOINT_DIR, FIGURES_DIR, LATENT_CH, LATENT_SCALE)
from src import dataset_final     as _dataset_final
from src import dataset_combined as _dataset_combined
from src.models       import Autoencoder3D, DenoisingUNet3D
from src.diffusion    import DiffusionSchedule
from src.conditioning import build_conditioner
from src.inference    import synthesize_tau_pet
from src.ema          import EMA


# ── Loss ─────────────────────────────────────────────────────────────────────

def ae_loss(recon, x, mean, logvar, kl_weight=1e-4):
    recon_loss = F.l1_loss(recon, x)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss


# ── Latent cache ──────────────────────────────────────────────────────────────

class LatentDataset(Dataset):
    """Dataset of pre-computed (z_pet, z_mri, cond) tuples stored in CPU RAM."""
    def __init__(self, z_pets, z_mris, conds):
        self.z_pets = z_pets
        self.z_mris = z_mris
        self.conds  = conds

    def __len__(self):
        return len(self.z_pets)

    def __getitem__(self, idx):
        return self.z_pets[idx], self.z_mris[idx], self.conds[idx]


def precompute_latents(ae, dataset, device, desc="Caching latents"):
    """Encode all volumes one-at-a-time and return (z_pets, z_mris, conds) as CPU tensors.

    Processing batch=1 at a time keeps peak VRAM low (~130 MB vs ~1 GB for batch=8).
    Total cache size for 179 subjects is ~11 MB — negligible.
    """
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
             val_loader=None, patience=20, val_every=5, figures_dir=None, kl_weight=1e-4,
             resume=False):
    ae_opt = torch.optim.Adam(ae.parameters(), lr=LR)
    scaler = GradScaler('cuda')
    train_losses = []
    val_epochs_list, val_losses_list = [], []
    best_val_loss  = float("inf")
    best_epoch     = 0
    epochs_no_improv = 0
    start_epoch    = 0
    best_ckpt_path = ckpt_path.replace(".pt", "_best.pt")

    # ── Resume AE training from a partial checkpoint (--resume-ae) ──────────────
    # Lets a paused/requeued AE fold pick up where it left off instead of restarting
    # at epoch 0. Restores weights, optimizer, AMP scaler, epoch, and loss history.
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        ae.load_state_dict(ck["ae"])
        if "ae_opt" in ck: ae_opt.load_state_dict(ck["ae_opt"])
        if "scaler" in ck: scaler.load_state_dict(ck["scaler"])
        start_epoch  = ck.get("epoch", 0)
        train_losses = ck.get("ae_losses", [])
        for ep, vl in ck.get("ae_val_losses", []):
            val_epochs_list.append(ep); val_losses_list.append(vl)
        if val_losses_list:
            best_val_loss = min(val_losses_list)
            best_epoch    = val_epochs_list[val_losses_list.index(best_val_loss)]
        print(f"Resuming AE from {ckpt_path} at epoch {start_epoch} "
              f"(best val={best_val_loss:.4f} @ ep {best_epoch})", flush=True)
    elif resume:
        print(f"--resume-ae set but no checkpoint at {ckpt_path}; training from scratch.", flush=True)

    print("=== Autoencoder Pretraining ===", flush=True)
    start = time.time()
    for epoch in range(start_epoch, n_epochs):
        ae.train()
        epoch_loss = 0.0
        t0 = time.time()
        for pet, mri, _ in train_loader:
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

        # ── Validation ────────────────────────────────────────────────────────
        val_str = ""
        if val_loader is not None and (epoch + 1) % val_every == 0:
            ae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for pet, mri, _ in val_loader:
                    pet, mri = pet.to(device), mri.to(device)
                    with autocast('cuda'):
                        for vol in (pet, mri):
                            recon, mean, logvar = ae(vol)
                            val_loss += ae_loss(recon, vol, mean, logvar, kl_weight=kl_weight).item()
            val_loss /= len(val_loader)
            val_epochs_list.append(epoch + 1)
            val_losses_list.append(val_loss)
            val_str = f"  val={val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch + 1
                epochs_no_improv = 0
                torch.save({"ae": ae.state_dict(), "ae_losses": train_losses,
                            "ae_val_losses": list(zip(val_epochs_list, val_losses_list)),
                            "latent_ch": ae.latent_ch,
                            "ae_scale": ae.scale, "ae_res_blocks": ae.n_res_blocks,
                            "epoch": epoch + 1}, best_ckpt_path)
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
                        "latent_ch": ae.latent_ch,
                        "ae_scale": ae.scale, "ae_res_blocks": ae.n_res_blocks,
                        "ae_opt": ae_opt.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch + 1}, ckpt_path)

    print(f"AE pretraining done. Total: {(time.time()-start)/60:.1f}m")

    # ── Loss curve ────────────────────────────────────────────────────────────
    train_x = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_x, train_losses, label="Train", alpha=0.8)
    if val_epochs_list:
        ax.plot(val_epochs_list, val_losses_list, "o-", lw=1.5, ms=4, label="Val")
        ax.axvline(x=best_epoch, color="g", ls="--", lw=1.5, label=f"Best val (ep {best_epoch})")
    ax.set(xlabel="Epoch", ylabel="Loss", title="AE Loss")
    ax.legend()
    fig.tight_layout()
    figs_out = figures_dir or FIGURES_DIR
    os.makedirs(figs_out, exist_ok=True)
    fig.savefig(os.path.join(figs_out, "ae_loss.png"), dpi=150)
    plt.close(fig)
    print(f"AE loss curve saved to {figs_out}/ae_loss.png")

    return train_losses


# ── SSIM monitoring ───────────────────────────────────────────────────────────

def _monitor_ssim(ae, unet, conditioner, schedule, latent_std,
                  val_ds, epoch, monitor_dir, device, n_steps=50):
    """Quick SSIM on one fixed val sample — 2D mid-axial slice only.

    Uses val_ds[0] every call so the metric is comparable across epochs.
    AE is needed here only for decode (batch=1, manageable VRAM).
    """
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

    mid = real_np.shape[2] // 2
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

def train_diffusion(ae, unet, conditioner, schedule, latent_std,
                    latent_train_loader, latent_val_loader, val_ds,
                    n_epochs, start_epoch, ckpt_path, device,
                    patience=50, val_every=1, monitor_every=25,
                    monitor_dir=None, figures_dir=None, lr=LR,
                    ema_decay=0.0, min_snr_gamma=0.0, ema_state=None):
    """Train the diffusion U-Net on pre-cached latents.

    latent_train_loader / latent_val_loader: yield (z0, zm, cond) — no AE on GPU.
    val_ds: original dataset kept for SSIM monitoring (needs AE decode, batch=1).
    latent_std: used for checkpoint saving and SSIM decode unscaling.

    ema_decay > 0 enables EMA of UNet weights (validated/monitored/saved as
    "unet_ema"; "unet" stays raw for resume). min_snr_gamma > 0 enables Min-SNR
    loss weighting on the TRAINING loss only — val loss stays plain MSE so it is
    comparable across runs. Both default OFF (identical to prior behavior).
    """
    VAL_SEED  = 0  # fixed seed for deterministic validation timesteps/noise
    trainable = list(unet.parameters()) + conditioner.trainable_parameters()
    diff_opt  = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    ema = EMA(unet, decay=ema_decay) if ema_decay and ema_decay > 0 else None
    if ema is not None and ema_state is not None:
        ema.load_state_dict(ema_state); ema.to(device)
    if ema is not None:
        print(f"EMA enabled (decay={ema_decay})", flush=True)
    if min_snr_gamma and min_snr_gamma > 0:
        print(f"Min-SNR weighting enabled (gamma={min_snr_gamma})", flush=True)

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
            # z0 and zm are already scaled by latent_std — no AE call needed
            z0, zm, cond_data = z0.to(device), zm.to(device), cond_data.to(device)

            B       = z0.shape[0]
            t       = torch.randint(0, T_STEPS, (B,), device=device)
            zt, eps = schedule.q_sample(z0, t)

            ctx     = conditioner.encode(cond_data)
            ht      = torch.cat([zt, zm], dim=1)
            eps_hat = unet(ht, t, ctx)
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

        avg_train = epoch_loss / len(latent_train_loader)
        losses.append(avg_train)
        elapsed   = time.time() - start
        remaining = (time.time() - t0) * (n_epochs - epoch - 1)

        # ── Validation loss ───────────────────────────────────────────────────
        val_str = ""
        if (epoch + 1) % val_every == 0:
            unet.eval()
            if hasattr(conditioner, "eval"):
                conditioner.eval()
            if ema is not None:
                ema.store(unet); ema.copy_to(unet)   # validate the EMA weights
            val_loss = 0.0
            # Deterministic validation: re-seed each pass so the timesteps and
            # noise are identical across epochs. Otherwise val loss is dominated
            # by which random t / noise happen to be drawn, making the curve too
            # noisy for reliable early stopping / best-checkpoint selection.
            val_gen = torch.Generator(device=device).manual_seed(VAL_SEED)
            with torch.no_grad():
                for z0, zm, cond_data in latent_val_loader:
                    z0, zm, cond_data = z0.to(device), zm.to(device), cond_data.to(device)
                    B       = z0.shape[0]
                    t       = torch.randint(0, T_STEPS, (B,), device=device,
                                            generator=val_gen)
                    eps     = torch.randn(z0.shape, device=device, generator=val_gen)
                    zt, eps = schedule.q_sample(z0, t, noise=eps)
                    ctx     = conditioner.encode(cond_data)
                    ht      = torch.cat([zt, zm], dim=1)
                    eps_hat = unet(ht, t, ctx)
                    val_loss += F.mse_loss(eps_hat, eps).item()
            val_loss /= len(latent_val_loader)
            if ema is not None:
                ema.restore(unet)                    # back to raw weights for training/saving
            val_epochs_list.append(epoch + 1)
            val_losses_list.append(val_loss)
            val_str = f"  val={val_loss:.6f}"

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_epoch       = epoch + 1
                epochs_no_improv = 0
                _save_diff_ckpt(best_ckpt_path, ae, unet, conditioner, latent_std,
                                losses, epoch + 1,
                                diff_val_losses=list(zip(val_epochs_list, val_losses_list)),
                                ema=ema,
                                noise_schedule=getattr(schedule, "schedule_name", "linear"))
            else:
                epochs_no_improv += val_every

        # ── SSIM monitor ──────────────────────────────────────────────────────
        ssim_str = ""
        if (epoch + 1) % monitor_every == 0:
            if ema is not None:
                ema.store(unet); ema.copy_to(unet)
            ssim_val = _monitor_ssim(ae, unet, conditioner, schedule, latent_std,
                                     val_ds, epoch + 1,
                                     monitor_dir or os.path.join(FIGURES_DIR, "monitor"),
                                     device)
            if ema is not None:
                ema.restore(unet)
            ssim_str = f"  SSIM={ssim_val:.3f}"

        # ── Logging & checkpoint ──────────────────────────────────────────────
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}"
                  f"  train={avg_train:.6f}{val_str}{ssim_str}"
                  f"  elapsed={elapsed/60:.1f}m  remaining={remaining/60:.1f}m",
                  flush=True)
            _save_diff_ckpt(ckpt_path, ae, unet, conditioner, latent_std,
                            losses, epoch + 1,
                            diff_val_losses=list(zip(val_epochs_list, val_losses_list)),
                            ema=ema,
                            noise_schedule=getattr(schedule, "schedule_name", "linear"))

        # ── Early stopping ────────────────────────────────────────────────────
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
        ax.axvline(x=best_epoch, color="g", ls="--", lw=1.5, label=f"Best val (ep {best_epoch})")
    ax.set(xlabel="Epoch", ylabel="MSE Loss", title=f"Diffusion Loss ({mode_tag})")
    ax.legend()
    fig.tight_layout()
    figs_out = figures_dir or FIGURES_DIR
    os.makedirs(figs_out, exist_ok=True)
    fig.savefig(os.path.join(figs_out, f"diff_loss_{mode_tag}.png"), dpi=150)
    plt.close(fig)

    return losses, list(zip(val_epochs_list, val_losses_list))


def _save_diff_ckpt(path, ae, unet, conditioner, latent_std, diff_losses, epoch,
                    diff_val_losses=None, ema=None, noise_schedule="linear"):
    ckpt = {
        "ae":             ae.state_dict(),
        "unet":           unet.state_dict(),   # raw weights (resume-safe)
        "conditioner":    conditioner.state_dict(),
        "latent_std":     latent_std.cpu(),
        "latent_ch":      ae.latent_ch,
        # AE architecture, so inference/eval can rebuild the exact encoder-decoder that was
        # trained (paper spec = 2 residual blocks per level). Absent on pre-2026-07 checkpoints.
        "ae_scale":       getattr(ae, "scale", LATENT_SCALE),
        "ae_res_blocks":  getattr(ae, "n_res_blocks", 1),
        "noise_schedule": noise_schedule,   # sampler MUST match this or samples corrupt
        "diff_losses":    diff_losses,
        "diff_val_losses": diff_val_losses or [],
        "epoch":          epoch,
    }
    if ema is not None:
        ckpt["unet_ema"] = ema.shadow           # EMA weights (preferred at eval/inference)
    torch.save(ckpt, path)


# ── Synthesis (inference) ─────────────────────────────────────────────────────

@torch.no_grad()
def synthesize(ae, unet, conditioner, schedule, mri_vol, cond_data, latent_std,
               n_steps=200, device=DEVICE):
    cond = cond_data.squeeze(0) if cond_data.dim() == 2 else cond_data
    return synthesize_tau_pet(
        mri_vol, cond, ae, unet, schedule, conditioner.encode, latent_std,
        device=device, n_steps=n_steps,
    )


# ── Evaluation helper ─────────────────────────────────────────────────────────

def quick_eval(ae, unet, conditioner, schedule, latent_std, test_ds,
               n_samples=10, mode="ptau217", device=DEVICE):
    maes, mses = [], []
    for i in range(min(n_samples, len(test_ds))):
        pet, mri, cond = test_ds[i]
        gen = synthesize_tau_pet(
            mri.unsqueeze(0), cond, ae, unet, schedule,
            conditioner.encode, latent_std, device=device, n_steps=50,
        )
        pet_np = pet.squeeze().numpy()
        gen_np = gen.squeeze().numpy()
        maes.append(np.abs(pet_np - gen_np).mean())
        mses.append(((pet_np - gen_np) ** 2).mean())
    print(f"Eval ({mode}, n={len(maes)}):  MAE={np.mean(maes):.4f}  MSE={np.mean(mses):.6f}")
    return np.mean(maes), np.mean(mses)


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train TauGenNet")
    p.add_argument("--mode",           choices=["ptau217", "ptau217_biobert", "ptau217_mlp", "atrophy", "combined"], required=True)
    p.add_argument("--ae-epochs",      type=int,  default=AE_EPOCHS)
    p.add_argument("--diff-epochs",    type=int,  default=DIFF_EPOCHS)
    p.add_argument("--batch-size",     type=int,  default=BATCH_SIZE)
    p.add_argument("--checkpoint-dir", type=str,  default=CHECKPOINT_DIR)
    p.add_argument("--ae-checkpoint",  type=str,  default=AE_CHECKPOINT_PATH)
    p.add_argument("--skip-ae",        action="store_true",
                   help="Load AE from --ae-checkpoint instead of retraining")
    p.add_argument("--resume-ae",      action="store_true",
                   help="Resume AE training from the partial --ae-checkpoint (restores "
                        "weights, optimizer, AMP scaler, epoch, loss history). For "
                        "pausing/requeuing AE folds. Falls back to fresh training if no "
                        "checkpoint exists. Default: off.")
    p.add_argument("--eval-only",      action="store_true",
                   help="Skip training; load checkpoint and run quick eval")
    p.add_argument("--patience",       type=int,  default=50)
    p.add_argument("--val-every",      type=int,  default=1)
    p.add_argument("--ae-val-every",   type=int,  default=1,
                   help="Validation cadence (epochs) for AE training only; "
                        "does not affect diffusion (see --val-every).")
    p.add_argument("--unet-channels",  type=str,  default="256,512,768",
                   help="UNet channel widths, e.g. '128,256,512'")
    p.add_argument("--n-transformer",  type=int,  default=3,
                   help="Transformer blocks per ResBlock in each UNet level (default 3)")
    p.add_argument("--monitor-every",  type=int,  default=25)
    p.add_argument("--split",          type=str,  default="64/16/20",
                   help="Train/val/test split percentages, e.g. '80/10/10'")
    p.add_argument("--monitor-dir",    type=str,  default=None,
                   help="Directory for epoch monitoring figures (default: FIGURES_DIR/monitor)")
    p.add_argument("--figures-dir",    type=str,  default=None,
                   help="Directory for loss curve figures (default: FIGURES_DIR)")
    p.add_argument("--use-mask",       action=argparse.BooleanOptionalAction, default=True,
                   help="Mask MRI/PET to DK atlas (T1_seg labels 1-86), zeroing only background "
                        "(label 0: skull/dura/ventricles). Default: on; use --no-use-mask to disable.")
    p.add_argument("--use-mentor-split", action=argparse.BooleanOptionalAction, default=True,
                   help="Use Anil's fixed mentor split (heldout_test_split.csv + train_val_split.csv): "
                        "fixed test set, CV rotates train/val over the dev pool. Default: on. "
                        "--no-use-mentor-split reverts to the legacy random split (must match how any "
                        "checkpoint being resumed/compared was trained).")
    p.add_argument("--fold",           type=int, default=None,
                   help="Fold index (0-indexed) for k-fold CV; uses fold-based dataset splitting")
    p.add_argument("--n-folds",        type=int, default=5,
                   help="Number of folds for k-fold CV (default 5)")
    p.add_argument("--latent-ch",      type=int, default=LATENT_CH,
                   help="Latent channel count (must match the AE checkpoint)")
    p.add_argument("--noise-schedule", choices=["linear", "cosine"], default="linear",
                   help="DDPM beta schedule. Persisted in the checkpoint; sampling MUST use the "
                        "same one or samples are silently corrupted.")
    p.add_argument("--latent-scale",   type=int, default=LATENT_SCALE, choices=[4, 8, 16],
                   help="AE downsampling factor = latent SPATIAL resolution. 96x112x96 -> "
                        "4:(24x28x24)  8:(12x14x12, default)  16:(6x7x6). Grid axis 1.")
    p.add_argument("--ae-res-blocks",  type=int, default=2,
                   help="AE depth: residual blocks per encoder/decoder level. Default 2 = the "
                        "TauGenNet paper spec ('four levels ... each containing two residual "
                        "blocks'). Use 1 to reproduce the pre-2026-07 architecture. Loading an "
                        "existing AE (--skip-ae) always uses that checkpoint's own value.")
    p.add_argument("--lr",             type=float, default=LR,
                   help="Learning rate for diffusion AdamW optimizer (default 1e-4)")
    p.add_argument("--kl-weight",      type=float, default=1e-4,
                   help="KL divergence weight in AE loss (default 1e-4)")
    p.add_argument("--ema-decay",      type=float, default=0.0,
                   help="EMA decay for the diffusion UNet weights; 0 = OFF (default). "
                        "Typical: 0.999. Eval/inference prefer EMA weights when saved.")
    p.add_argument("--min-snr-gamma",  type=float, default=0.0,
                   help="Min-SNR-gamma loss weighting (eps-prediction form); 0 = OFF "
                        "(uniform, default). Typical: 5.")
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
                train_frac=train_frac, val_frac=val_frac,
                use_mentor_split=args.use_mentor_split, **fold_kwargs,
            )
    else:
        # ptau217_mlp / ptau217_biobert use the same data as ptau217
        # (1-dim scalar conditioning; only the text/MLP encoder differs)
        dataset_mode = "ptau217" if args.mode in ("ptau217_mlp", "ptau217_biobert") else args.mode
        train_ds, val_ds, test_ds, train_loader, val_loader, _ = \
            _dataset_final.build_dataloaders(
                mode=dataset_mode, batch_size=args.batch_size, use_dk_mask=args.use_mask,
                train_frac=train_frac, val_frac=val_frac,
                use_mentor_split=args.use_mentor_split, **fold_kwargs,
            )
    print(f"Mode: {args.mode}  |  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ── Models ────────────────────────────────────────────────────────────────
    # When skipping AE training, rebuild the AE at the CHECKPOINT's architecture, not the CLI
    # default — older checkpoints predate ae_scale/ae_res_blocks, so fall back to the original
    # 8x / 1-resblock spec. Without this, the new --ae-res-blocks default (2) would fail to load them.
    ae_scale, ae_rb = args.latent_scale, args.ae_res_blocks
    if (args.skip_ae or getattr(args, "eval_only", False)) and os.path.exists(args.ae_checkpoint):
        _peek    = torch.load(args.ae_checkpoint, map_location="cpu")
        ae_scale = _peek.get("ae_scale", LATENT_SCALE)
        ae_rb    = _peek.get("ae_res_blocks", 1)
        del _peek
        print(f"AE arch taken from checkpoint: scale={ae_scale}  res_blocks/level={ae_rb}")
    ae          = Autoencoder3D(latent_ch=args.latent_ch, scale=ae_scale,
                                n_res_blocks=ae_rb).to(device)
    print(f"AE: latent_ch={args.latent_ch}  scale={ae_scale} "
          f"(latent {96//ae_scale}x{112//ae_scale}x{96//ae_scale})  "
          f"res_blocks/level={ae_rb}")
    print(f"AE params:   {sum(p.numel() for p in ae.parameters()):,}")

    # Only create UNet/conditioner/schedule if doing diffusion training
    conditioner = None
    unet = None
    schedule = None
    if args.diff_epochs > 0:
        unet_ch = tuple(int(x) for x in args.unet_channels.split(","))
        conditioner = build_conditioner(args.mode, device=device)
        unet        = DenoisingUNet3D(latent_ch=args.latent_ch,
                                      context_dim=conditioner.out_dim,
                                      ch_list=unet_ch,
                                      n_transformer=args.n_transformer).to(device)
        schedule    = DiffusionSchedule(device=device, schedule=args.noise_schedule)
        print(f"Noise schedule: {args.noise_schedule}")
        print(f"UNet params: {sum(p.numel() for p in unet.parameters()):,}")
        extra = sum(p.numel() for p in conditioner.trainable_parameters())
        if extra:
            print(f"Conditioner params: {extra:,}")

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
        ae_losses = train_ae(ae, train_loader, args.ae_epochs, args.ae_checkpoint, device,
                             val_loader=val_loader, patience=args.patience,
                             val_every=args.ae_val_every, figures_dir=args.figures_dir,
                             kl_weight=args.kl_weight, resume=args.resume_ae)

    # AE-only run: skip everything below
    if args.diff_epochs == 0:
        print("--diff-epochs 0: skipping diffusion training.")
        return

    if args.eval_only:
        if os.path.exists(diff_ckpt):
            ckpt = torch.load(diff_ckpt, map_location=device)
            unet.load_state_dict(ckpt["unet"])
            if ckpt.get("conditioner"):
                conditioner.load_state_dict(ckpt["conditioner"])
            latent_std = ckpt.get("latent_std", torch.ones(1, args.latent_ch, 1, 1, 1)).to(device)
            print(f"Loaded diffusion model from {diff_ckpt}")
        else:
            latent_std = torch.ones(1, args.latent_ch, 1, 1, 1, device=device)
        quick_eval(ae, unet, conditioner, schedule, latent_std,
                   test_ds, mode=args.mode, device=device)
        return

    # ── Load or resume diffusion ──────────────────────────────────────────────
    start_epoch     = 0
    diff_losses     = []
    diff_val_losses = []
    latent_std      = None
    ema_state       = None

    if os.path.exists(diff_ckpt):
        ckpt = torch.load(diff_ckpt, map_location=device)
        ae.load_state_dict(ckpt["ae"])
        unet.load_state_dict(ckpt["unet"])
        ema_state = ckpt.get("unet_ema", None)   # resume EMA shadow if present
        if ckpt.get("conditioner"):
            conditioner.load_state_dict(ckpt["conditioner"])
        latent_std  = ckpt.get("latent_std", None)
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

    # Compute latent_std from cached raw (unscaled) train latents if not loaded
    if latent_std is None:
        z_all      = torch.cat([z_pets_tr, z_mris_tr], dim=0)
        latent_std = z_all.std(dim=(0, 2, 3, 4), keepdim=True).to(device)
        print(f"latent_std per channel: {latent_std.squeeze().tolist()}")

    # Scale cached latents
    ls_cpu       = latent_std.cpu()
    z_pets_tr   /= ls_cpu;  z_mris_tr  /= ls_cpu
    z_pets_val  /= ls_cpu;  z_mris_val /= ls_cpu

    # ── Build latent DataLoaders ──────────────────────────────────────────────
    lat_train_ds  = LatentDataset(z_pets_tr,  z_mris_tr,  conds_tr)
    lat_val_ds    = LatentDataset(z_pets_val, z_mris_val, conds_val)
    lat_train_ldr = DataLoader(lat_train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0)
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
        ema_decay=args.ema_decay,
        min_snr_gamma=args.min_snr_gamma,
        ema_state=ema_state,
    )
    diff_losses     += new_losses
    diff_val_losses += new_val_losses

    # ── Loss plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ae_losses)
    axes[0].set(title="AE Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(diff_losses, label="Train", alpha=0.8)
    if diff_val_losses:
        ve, vl = zip(*diff_val_losses)
        axes[1].plot(ve, vl, "o-", ms=3, lw=1.2, label="Val")
    axes[1].legend()
    axes[1].set(title=f"Diffusion Loss ({args.mode})", xlabel="Epoch", ylabel="MSE")
    plt.tight_layout()
    out_fig = os.path.join(args.checkpoint_dir, f"losses_{args.mode}.png")
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Saved loss plot → {out_fig}")


if __name__ == "__main__":
    main()
