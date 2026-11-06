#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
train_dynamic_prompt.py – Train TauGenNet with CoMA-style dynamic prompt head (Option 1).

The base UNet input is unchanged (in_ch=6, [z_t, z_m]). After the UNet forward pass
a learned dynamic prompt is modulated by the atlas-based atrophy map and applied as a
post-UNet correction in latent space.

Prompt selection per sample:
  diagnosis=1 (AD)  → pos_prompt
  diagnosis=0 (MCI) → neg_prompt

Diagnosis is derived from the subject folder name (1mm_parcellated_AD_subj / MCI_subj).
Cross-attention conditioning (atrophy_vec → 16 tokens) is kept in parallel.

Usage
-----
python scripts/train_dynamic_prompt.py --skip-ae \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_paper.pt
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
                        AE_CHECKPOINT_PATH, FIGURES_DIR, LATENT_CH)
from src import dataset_spatial as _dataset
from src.models                import Autoencoder3D    # standard AE; in_ch=1 unchanged
from src.models_dynamic_prompt import DenoisingUNet3DWithPrompt
from src.diffusion             import DiffusionSchedule
from src.conditioning          import AtrophyConditioner

_DEFAULT_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "checkpoints", "dynamic_prompt")


# ── Loss ──────────────────────────────────────────────────────────────────────

def ae_loss(recon, x, mean, logvar, kl_weight=1e-4):
    recon_loss = F.l1_loss(recon, x)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss


# ── Latent cache ──────────────────────────────────────────────────────────────

class DynamicLatentDataset(Dataset):
    """Cached (z_pet, z_mri, atrophy_vec, atrophy_map_latent, diagnosis) tuples."""
    def __init__(self, z_pets, z_mris, atrophy_vecs, atrophy_maps_latent, diagnoses):
        self.z_pets              = z_pets
        self.z_mris              = z_mris
        self.atrophy_vecs        = atrophy_vecs
        self.atrophy_maps_latent = atrophy_maps_latent
        self.diagnoses           = diagnoses

    def __len__(self):
        return len(self.z_pets)

    def __getitem__(self, idx):
        return (self.z_pets[idx], self.z_mris[idx], self.atrophy_vecs[idx],
                self.atrophy_maps_latent[idx], self.diagnoses[idx])


def precompute_latents(ae, dataset, device, desc="Caching latents"):
    ae.eval()
    z_pets, z_mris, atrophy_vecs, atrophy_maps_latent, diagnoses = [], [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=desc, leave=False):
            pet, mri, atrophy_vec, atrophy_map, diagnosis = dataset[i]
            z_pets.append(ae.encode_mean(pet.unsqueeze(0).to(device)).cpu().squeeze(0))
            z_mris.append(ae.encode_mean(mri.unsqueeze(0).to(device)).cpu().squeeze(0))
            atrophy_vecs.append(atrophy_vec)
            am_latent = F.avg_pool3d(atrophy_map.unsqueeze(0), kernel_size=8).squeeze(0)
            atrophy_maps_latent.append(am_latent)
            diagnoses.append(diagnosis)
    return (torch.stack(z_pets), torch.stack(z_mris), torch.stack(atrophy_vecs),
            torch.stack(atrophy_maps_latent), torch.stack(diagnoses))


# ── AE pretraining ────────────────────────────────────────────────────────────

def train_ae(ae, train_loader, n_epochs, ckpt_path, device,
             val_loader=None, patience=20, val_every=5, figures_dir=None, kl_weight=1e-4):
    ae_opt = torch.optim.Adam(ae.parameters(), lr=LR)
    scaler = GradScaler('cuda')
    train_losses, val_epochs_list, val_losses_list = [], [], []
    best_val_loss    = float("inf")
    best_epoch       = 0
    epochs_no_improv = 0
    best_ckpt_path   = ckpt_path.replace(".pt", "_best.pt")

    print("=== Autoencoder Pretraining ===", flush=True)
    start = time.time()
    for epoch in range(n_epochs):
        ae.train()
        epoch_loss = 0.0
        t0 = time.time()
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
                best_val_loss    = val_loss
                best_epoch       = epoch + 1
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
                  val_ds, epoch, monitor_dir, device, n_steps=50):
    """Quick SSIM on val_ds[0] using local DDPM loop (passes atrophy_map + diagnosis)."""
    unet.eval(); ae.eval()
    if hasattr(conditioner, "eval"):
        conditioner.eval()

    pet, mri, atrophy_vec, atrophy_map, diagnosis = val_ds[0]
    real_np = pet.squeeze().numpy()

    mri_b     = mri.unsqueeze(0).to(device)
    zm        = ae.encode_mean(mri_b) / latent_std
    am_latent = F.avg_pool3d(atrophy_map.unsqueeze(0).to(device), kernel_size=8)
    ctx       = conditioner.encode(atrophy_vec.unsqueeze(0).to(device))
    diag_b    = diagnosis.unsqueeze(0).to(device)
    zt        = torch.randn_like(zm)

    step_indices = list(range(0, T_STEPS, T_STEPS // n_steps))[::-1]
    for t_idx in step_indices:
        t_batch = torch.full((1,), t_idx, device=device, dtype=torch.long)
        ht      = torch.cat([zt, zm], dim=1)   # (1, 6, lH, lW, lD)
        eps_hat = unet(ht, t_batch, ctx, atrophy_map=am_latent, diagnosis=diag_b)
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
    axes[0].imshow(real_np[:, :, mid], cmap="hot", vmin=0, vmax=1); axes[0].set_title("Real PET"); axes[0].axis("off")
    axes[1].imshow(gen_np[:, :, mid], cmap="hot", vmin=0, vmax=1); axes[1].set_title(f"Generated (ep {epoch})"); axes[1].axis("off")
    axes[2].imshow(np.abs(real_np[:, :, mid] - gen_np[:, :, mid]), cmap="coolwarm", vmin=0, vmax=0.5)
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
               diff_val_losses=None):
    torch.save({
        "ae":             ae.state_dict(),
        "unet":           unet.state_dict(),
        "conditioner":    conditioner.state_dict(),
        "latent_std":     latent_std.cpu(),
        "latent_ch":      ae.latent_ch,
        "diff_losses":    diff_losses,
        "diff_val_losses": diff_val_losses or [],
        "epoch":          epoch,
    }, path)


def train_diffusion(ae, unet, conditioner, schedule, latent_std,
                    lat_train_ldr, lat_val_ldr, val_ds,
                    n_epochs, start_epoch, ckpt_path, device,
                    patience=50, val_every=1, monitor_every=25,
                    monitor_dir=None, figures_dir=None, lr=LR):
    # Include prompt parameters in the optimizer
    prompt_params = [p for name, p in unet.named_parameters()
                     if any(k in name for k in ("pos_prompt", "neg_prompt",
                                                "general_prompt", "deep_mod",
                                                "fusion", "pred_head"))]
    base_params   = [p for name, p in unet.named_parameters()
                     if not any(k in name for k in ("pos_prompt", "neg_prompt",
                                                     "general_prompt", "deep_mod",
                                                     "fusion", "pred_head"))]
    trainable = (
        [{"params": base_params,   "lr": lr}] +
        [{"params": prompt_params, "lr": lr * 5}] +   # higher LR for new prompt params
        [{"params": conditioner.trainable_parameters(), "lr": lr}]
    )
    diff_opt = torch.optim.AdamW(trainable, weight_decay=1e-4)

    losses, val_epochs_list, val_losses_list = [], [], []
    best_val_loss    = float("inf")
    best_epoch       = start_epoch
    epochs_no_improv = 0
    best_ckpt_path   = ckpt_path.replace(".pt", "_best.pt")
    ae.eval()

    print(f"=== Dynamic Prompt Diffusion Training (start={start_epoch}, patience={patience}) ===",
          flush=True)
    start = time.time()

    for epoch in range(start_epoch, n_epochs):
        unet.train()
        if hasattr(conditioner, "train"):
            conditioner.train()
        epoch_loss = 0.0
        t0 = time.time()

        for z0, zm, atrophy_vec, am_latent, diagnosis in lat_train_ldr:
            z0, zm        = z0.to(device), zm.to(device)
            atrophy_vec   = atrophy_vec.to(device)
            am_latent     = am_latent.to(device)      # (B, 1, 12, 14, 12)
            diagnosis     = diagnosis.to(device)       # (B,) long

            B       = z0.shape[0]
            t       = torch.randint(0, T_STEPS, (B,), device=device)
            zt, eps = schedule.q_sample(z0, t)

            ctx     = conditioner.encode(atrophy_vec)
            ht      = torch.cat([zt, zm], dim=1)       # (B, 6, 12, 14, 12)
            eps_hat = unet(ht, t, ctx, atrophy_map=am_latent, diagnosis=diagnosis)
            loss    = F.mse_loss(eps_hat, eps)

            diff_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            diff_opt.step()
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
            val_loss = 0.0
            with torch.no_grad():
                for z0, zm, atrophy_vec, am_latent, diagnosis in lat_val_ldr:
                    z0, zm      = z0.to(device), zm.to(device)
                    atrophy_vec = atrophy_vec.to(device)
                    am_latent   = am_latent.to(device)
                    diagnosis   = diagnosis.to(device)
                    B           = z0.shape[0]
                    t           = torch.randint(0, T_STEPS, (B,), device=device)
                    zt, eps     = schedule.q_sample(z0, t)
                    ctx         = conditioner.encode(atrophy_vec)
                    ht          = torch.cat([zt, zm], dim=1)
                    eps_hat     = unet(ht, t, ctx, atrophy_map=am_latent, diagnosis=diagnosis)
                    val_loss   += F.mse_loss(eps_hat, eps).item()
            val_loss /= len(lat_val_ldr)
            val_epochs_list.append(epoch + 1)
            val_losses_list.append(val_loss)
            val_str = f"  val={val_loss:.6f}"

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_epoch       = epoch + 1
                epochs_no_improv = 0
                _save_ckpt(best_ckpt_path, ae, unet, conditioner, latent_std,
                           losses, epoch + 1,
                           diff_val_losses=list(zip(val_epochs_list, val_losses_list)))
            else:
                epochs_no_improv += val_every

        ssim_str = ""
        if (epoch + 1) % monitor_every == 0:
            ssim_val = _monitor_ssim(ae, unet, conditioner, schedule, latent_std,
                                     val_ds, epoch + 1,
                                     monitor_dir or os.path.join(FIGURES_DIR, "dynamic_prompt", "monitor"),
                                     device)
            ssim_str = f"  SSIM={ssim_val:.3f}"

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}"
                  f"  train={avg_train:.6f}{val_str}{ssim_str}"
                  f"  elapsed={elapsed/60:.1f}m  remaining={remaining/60:.1f}m", flush=True)
            _save_ckpt(ckpt_path, ae, unet, conditioner, latent_std,
                       losses, epoch + 1,
                       diff_val_losses=list(zip(val_epochs_list, val_losses_list)))

        if epochs_no_improv >= patience:
            print(f"Early stopping at epoch {epoch+1} (best val={best_val_loss:.6f})")
            break

    print(f"Diffusion done. Total: {(time.time()-start)/60:.1f}m")
    return losses, list(zip(val_epochs_list, val_losses_list))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train TauGenNet — dynamic prompt option")
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
    p.add_argument("--atlas-path",     type=str,   default=None)
    p.add_argument("--fold",           type=int,   default=None)
    p.add_argument("--n-folds",        type=int,   default=5)
    p.add_argument("--latent-ch",      type=int,   default=LATENT_CH,
                   help="Latent channel count (must match the AE checkpoint)")
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--kl-weight",      type=float, default=1e-4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = DEVICE
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    diff_ckpt = os.path.join(args.checkpoint_dir, "diff_dynamic_prompt.pt")

    split_parts = [int(x) for x in args.split.split("/")]
    train_frac, val_frac = split_parts[0] / 100, split_parts[1] / 100
    fold_kwargs = {} if args.fold is None else {"fold_idx": args.fold, "n_folds": args.n_folds}

    train_ds, val_ds, test_ds, train_loader, val_loader, _ = _dataset.build_dataloaders(
        batch_size=args.batch_size, use_dk_mask=args.use_mask,
        train_frac=train_frac, val_frac=val_frac,
        atlas_path=args.atlas_path, **fold_kwargs,
    )
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    ae = Autoencoder3D(latent_ch=args.latent_ch).to(device)
    print(f"AE params: {sum(p.numel() for p in ae.parameters()):,}")

    conditioner = None
    unet        = None
    schedule    = None
    if args.diff_epochs > 0 or args.eval_only:
        unet_ch     = tuple(int(x) for x in args.unet_channels.split(","))
        conditioner = AtrophyConditioner().to(device)
        unet        = DenoisingUNet3DWithPrompt(
                          latent_ch=args.latent_ch,
                          ch_list=unet_ch, n_transformer=args.n_transformer
                      ).to(device)
        schedule    = DiffusionSchedule(device=device)
        n_base     = sum(p.numel() for n, p in unet.named_parameters()
                         if not any(k in n for k in ("pos_prompt","neg_prompt",
                                                      "general_prompt","deep_mod",
                                                      "fusion","pred_head")))
        n_prompt   = sum(p.numel() for n, p in unet.named_parameters()
                         if any(k in n for k in ("pos_prompt","neg_prompt",
                                                  "general_prompt","deep_mod",
                                                  "fusion","pred_head")))
        print(f"UNet base params: {n_base:,}  |  prompt head params: {n_prompt:,}")
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
    if os.path.exists(diff_ckpt):
        ckpt = torch.load(diff_ckpt, map_location=device)
        ae.load_state_dict(ckpt["ae"])
        unet.load_state_dict(ckpt["unet"])
        if ckpt.get("conditioner"):
            conditioner.load_state_dict(ckpt["conditioner"])
        latent_std  = ckpt.get("latent_std")
        if latent_std is not None:
            latent_std = latent_std.to(device)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resuming from epoch {start_epoch}")

    if args.eval_only:
        print("eval-only mode — run scripts/evaluate_final.py for full metrics")
        return

    # ── Pre-cache latents ─────────────────────────────────────────────────────
    print("Pre-computing latents (train)…")
    z_pets_tr, z_mris_tr, atrophy_vecs_tr, am_latents_tr, diagnoses_tr = precompute_latents(
        ae, train_ds, device, "Caching train"
    )
    print("Pre-computing latents (val)…")
    z_pets_val, z_mris_val, atrophy_vecs_val, am_latents_val, diagnoses_val = precompute_latents(
        ae, val_ds, device, "Caching val"
    )

    if latent_std is None:
        z_all      = torch.cat([z_pets_tr, z_mris_tr], dim=0)
        latent_std = z_all.std(dim=(0, 2, 3, 4), keepdim=True).to(device)
        print(f"latent_std: {latent_std.squeeze().tolist()}")

    ls_cpu = latent_std.cpu()
    z_pets_tr  /= ls_cpu;  z_mris_tr  /= ls_cpu
    z_pets_val /= ls_cpu;  z_mris_val /= ls_cpu

    lat_train_ds  = DynamicLatentDataset(z_pets_tr,  z_mris_tr,  atrophy_vecs_tr,
                                          am_latents_tr,  diagnoses_tr)
    lat_val_ds    = DynamicLatentDataset(z_pets_val, z_mris_val, atrophy_vecs_val,
                                          am_latents_val, diagnoses_val)
    lat_train_ldr = DataLoader(lat_train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=0)
    lat_val_ldr   = DataLoader(lat_val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=0)

    # ── Train ─────────────────────────────────────────────────────────────────
    figs_dir = args.figures_dir or os.path.join(FIGURES_DIR, "dynamic_prompt")
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
    )


if __name__ == "__main__":
    main()
