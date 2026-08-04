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

from src.config_relu import (DEVICE, T_STEPS, LR, AE_EPOCHS, DIFF_EPOCHS, BATCH_SIZE,
                             AE_CHECKPOINT_PATH, CHECKPOINT_DIR, FIGURES_DIR, LATENT_CH)
from src.dataset      import build_dataloaders
from src import dataset_v2_relu    as _dataset_v2
from src import dataset_combined_relu as _dataset_combined
from src.models_relu  import Autoencoder3D, DenoisingUNet3D
from src.diffusion    import DiffusionSchedule
from src.conditioning import build_conditioner
from src.inference    import synthesize_tau_pet


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

def train_ae(ae, train_loader, n_epochs, ckpt_path, device, kl_weight=1e-4):
    ae_opt = torch.optim.Adam(ae.parameters(), lr=LR)
    scaler = GradScaler('cuda')
    losses = []

    print("=== Autoencoder Pretraining ===", flush=True)
    start = time.time()
    for epoch in range(n_epochs):
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
        losses.append(avg)

        if (epoch + 1) % 5 == 0:
            elapsed   = time.time() - start
            remaining = (time.time() - t0) * (n_epochs - epoch - 1)
            print(f"  AE {epoch+1:3d}/{n_epochs}  loss={avg:.4f}  "
                  f"elapsed={elapsed/60:.1f}m  remaining={remaining/60:.1f}m", flush=True)
            torch.save({"ae": ae.state_dict(), "ae_losses": losses}, ckpt_path)

    print(f"AE pretraining done. Total: {(time.time()-start)/60:.1f}m")
    return losses


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
                    monitor_dir=None, figures_dir=None, lr=LR):
    """Train the diffusion U-Net on pre-cached latents.

    latent_train_loader / latent_val_loader: yield (z0, zm, cond) — no AE on GPU.
    val_ds: original dataset kept for SSIM monitoring (needs AE decode, batch=1).
    latent_std: used for checkpoint saving and SSIM decode unscaling.
    """
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
            # z0 and zm are already scaled by latent_std — no AE call needed
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

        # ── Validation loss ───────────────────────────────────────────────────
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

        # ── SSIM monitor ──────────────────────────────────────────────────────
        ssim_str = ""
        if (epoch + 1) % monitor_every == 0:
            ssim_val = _monitor_ssim(ae, unet, conditioner, schedule, latent_std,
                                     val_ds, epoch + 1,
                                     monitor_dir or os.path.join(FIGURES_DIR, "monitor"),
                                     device)
            ssim_str = f"  SSIM={ssim_val:.3f}"

        # ── Logging & checkpoint ──────────────────────────────────────────────
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}"
                  f"  train={avg_train:.6f}{val_str}{ssim_str}"
                  f"  elapsed={elapsed/60:.1f}m  remaining={remaining/60:.1f}m",
                  flush=True)
            _save_diff_ckpt(ckpt_path, ae, unet, conditioner, latent_std,
                            losses, epoch + 1,
                            diff_val_losses=list(zip(val_epochs_list, val_losses_list)))

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
                    diff_val_losses=None):
    torch.save({
        "ae":             ae.state_dict(),
        "unet":           unet.state_dict(),
        "conditioner":    conditioner.state_dict(),
        "latent_std":     latent_std.cpu(),
        "diff_losses":    diff_losses,
        "diff_val_losses": diff_val_losses or [],
        "epoch":          epoch,
    }, path)


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
    p.add_argument("--mode",           choices=["ptau217", "atrophy", "combined"], required=True)
    p.add_argument("--ae-epochs",      type=int,  default=AE_EPOCHS)
    p.add_argument("--diff-epochs",    type=int,  default=DIFF_EPOCHS)
    p.add_argument("--batch-size",     type=int,  default=BATCH_SIZE)
    p.add_argument("--checkpoint-dir", type=str,  default=CHECKPOINT_DIR)
    p.add_argument("--ae-checkpoint",  type=str,  default=AE_CHECKPOINT_PATH)
    p.add_argument("--skip-ae",        action="store_true",
                   help="Load AE from --ae-checkpoint instead of retraining")
    p.add_argument("--eval-only",      action="store_true",
                   help="Skip training; load checkpoint and run quick eval")
    p.add_argument("--patience",       type=int,  default=50)
    p.add_argument("--val-every",      type=int,  default=1)
    p.add_argument("--unet-channels",  type=str,  default="256,512,768",
                   help="UNet channel widths, e.g. '128,256,512'")
    p.add_argument("--n-transformer",  type=int,  default=3,
                   help="Transformer blocks per ResBlock in each UNet level (default 3)")
    p.add_argument("--monitor-every",  type=int,  default=25)
    p.add_argument("--dataset",        choices=["v1", "v2"], default="v1",
                   help="v1=original 72/11/28 split; v2=70/10/20 split (dataset_final.py)")
    p.add_argument("--monitor-dir",    type=str,  default=None,
                   help="Directory for epoch monitoring figures (default: FIGURES_DIR/monitor)")
    p.add_argument("--use-mask",       action=argparse.BooleanOptionalAction, default=True,
                   help="Mask MRI/PET to DK atlas (T1_seg labels 1-86), zeroing only background "
                        "(label 0: skull/dura/ventricles). Default: on; use --no-use-mask to disable.")
    p.add_argument("--lr",             type=float, default=LR,
                   help="Learning rate for diffusion AdamW optimizer (default 1e-4)")
    p.add_argument("--kl-weight",      type=float, default=1e-4,
                   help="KL divergence weight in AE loss (default 1e-4)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = DEVICE
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    diff_ckpt = os.path.join(args.checkpoint_dir, f"diff_{args.mode}.pt")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.mode == "combined":
        train_ds, val_ds, test_ds, train_loader, val_loader, _ = \
            _dataset_combined.build_dataloaders(mode=args.mode, batch_size=args.batch_size, use_dk_mask=args.use_mask)
    elif args.dataset == "v2":
        train_ds, val_ds, test_ds, train_loader, val_loader, _ = \
            _dataset_final.build_dataloaders(mode=args.mode, batch_size=args.batch_size, use_dk_mask=args.use_mask)
    else:
        train_ds, val_ds, test_ds, train_loader, val_loader, _ = \
            build_dataloaders(mode=args.mode, batch_size=args.batch_size, use_mask=args.use_mask)
    print(f"Mode: {args.mode}  |  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ── Models ────────────────────────────────────────────────────────────────
    ae          = Autoencoder3D().to(device)
    print(f"AE params:   {sum(p.numel() for p in ae.parameters()):,}")

    # Only create UNet/conditioner/schedule if doing diffusion training
    conditioner = None
    unet = None
    schedule = None
    if args.diff_epochs > 0:
        unet_ch = tuple(int(x) for x in args.unet_channels.split(","))
        conditioner = build_conditioner(args.mode, device=device)
        unet        = DenoisingUNet3D(context_dim=conditioner.out_dim,
                                      ch_list=unet_ch,
                                      n_transformer=args.n_transformer).to(device)
        schedule    = DiffusionSchedule(device=device)
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
                             kl_weight=args.kl_weight)

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
            latent_std = ckpt.get("latent_std", torch.ones(1, LATENT_CH, 1, 1, 1)).to(device)  # was 4
            print(f"Loaded diffusion model from {diff_ckpt}")
        else:
            latent_std = torch.ones(1, LATENT_CH, 1, 1, 1, device=device)  # was 4
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
        figures_dir=getattr(args, "figures_dir", None),
        lr=args.lr,
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
