"""
quick_figs.py — Generate Fig 3 (training curves) and Fig 4 (plasma sweep)
without a GPU. Runs on a login node.

Usage:
    python scripts/quick_figs.py --mode ptau217
    python scripts/quick_figs.py --mode atrophy
    python scripts/quick_figs.py --mode combined
    python scripts/quick_figs.py --mode all

    --skip-fig4   Only generate Fig 3 (instant, no inference needed)
    --n-steps N   DDIM steps for Fig 4 (default 20, lower = faster)
"""

import argparse, os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import VOL_SHAPE
from src.diffusion import DiffusionSchedule
from src.inference import load_models, synthesize_tau_pet

PLASMA_SWEEP_VALS = [0.65, 3.65, 6.65, 10.65]
H, W, D = VOL_SHAPE


# ── Fig 3: Training loss curve ─────────────────────────────────────────────────

def fig_3_training_curves(diff_losses, mode, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(diff_losses, color="#4C72B0", linewidth=1.2, label="Train MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"Training MSE — {mode} (paper arch)\n"
                 "(validation loss not available from checkpoint)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved Fig 3 → {save_path}")


# ── Fig 4: Plasma sweep ────────────────────────────────────────────────────────

def synthesize_at_plasma(mri_t, base_cond, plasma_val, ae, unet, schedule,
                          encode_cond, latent_std, mode, n_steps, device):
    cond = base_cond.clone()
    if mode == "combined":
        cond[86] = plasma_val
    else:
        cond = torch.tensor([plasma_val], dtype=torch.float32)
    return synthesize_tau_pet(
        mri_t, cond, ae, unet, schedule, encode_cond, latent_std,
        device=device, n_steps=n_steps, sampler="ddim"
    ).squeeze().numpy()


def fig_4_plasma_sweep(mri_t, base_cond, ae, unet, schedule, encode_cond,
                        latent_std, mode, save_path, n_steps, device):
    torch.manual_seed(0)
    print(f"Synthesising {len(PLASMA_SWEEP_VALS)} volumes at {n_steps} DDIM steps on {device}...")
    vols = []
    for i, pv in enumerate(PLASMA_SWEEP_VALS):
        print(f"  [{i+1}/{len(PLASMA_SWEEP_VALS)}] plasma = {pv}")
        vols.append(synthesize_at_plasma(mri_t, base_cond, pv, ae, unet, schedule,
                                          encode_cond, latent_std, mode, n_steps, device))

    view_labels = ["Axial view", "Sagittal view", "Coronal view"]
    fig, axes = plt.subplots(3, len(PLASMA_SWEEP_VALS),
                              figsize=(len(PLASMA_SWEEP_VALS) * 2.8, 3 * 2.5))

    for col, (vol, pv) in enumerate(zip(vols, PLASMA_SWEEP_VALS)):
        h, w, d = vol.shape
        slices = [vol[:, :, d // 2],
                  vol[:, w // 2, :],
                  vol[h // 2, :, :]]
        for row, sl in enumerate(slices):
            im = axes[row, col].imshow(sl, cmap="hot", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[0, col].set_title(f"Plasma is {pv}", fontsize=10, fontweight="bold")

    for row, lbl in enumerate(view_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=9)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="SUVR (norm.)")
    fig.suptitle(f"Generated tau PET across plasma p-tau217 values — {mode}",
                 fontsize=11, y=1.01)
    plt.subplots_adjust(left=0.10, right=0.90, hspace=0.05, wspace=0.05)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved Fig 4 → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_mode(mode, skip_fig4, n_steps):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    ckpt    = f"results/checkpoints/{mode}_paper/diff_{mode}.pt"
    fig_dir = f"results/figures/{mode}/paper"
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Loading checkpoint: {ckpt}")
    ae, unet, conditioner, latent_std, diff_losses = load_models(ckpt, mode, device=device)
    encode_cond = conditioner.encode

    # Fig 3 — instant, no inference
    fig_3_training_curves(diff_losses, mode,
                          os.path.join(fig_dir, "fig3_training_curve.pdf"))

    if skip_fig4 or mode == "atrophy":
        if mode == "atrophy":
            print("Fig 4 skipped: atrophy mode has no plasma conditioning.")
        return

    # Fig 4 — needs one test subject's MRI
    from src.dataset import build_dataloaders
    from src import dataset_combined as _dc
    if mode == "combined":
        _, _, _, _, _, test_loader = _dc.build_dataloaders(mode=mode)
    else:
        _, _, _, _, _, test_loader = build_dataloaders(mode=mode)

    schedule = DiffusionSchedule()
    for pet_b, mri_b, cond_b in test_loader:
        first_mri  = mri_b[0:1]
        first_cond = cond_b[0]
        break

    fig_4_plasma_sweep(first_mri, first_cond, ae, unet, schedule, encode_cond,
                       latent_std, mode,
                       os.path.join(fig_dir, "fig4_plasma_sweep.pdf"),
                       n_steps=n_steps, device=device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["atrophy", "ptau217", "combined", "all"],
                   default="ptau217")
    p.add_argument("--skip-fig4", action="store_true",
                   help="Only generate Fig 3 (no inference needed)")
    p.add_argument("--n-steps", type=int, default=20,
                   help="DDIM steps for Fig 4 (default 20, lower = faster)")
    args = p.parse_args()

    modes = ["atrophy", "ptau217", "combined"] if args.mode == "all" else [args.mode]
    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        run_mode(mode, skip_fig4=args.skip_fig4, n_steps=args.n_steps)

    print("\nDone.")
