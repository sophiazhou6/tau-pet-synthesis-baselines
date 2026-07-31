"""
paper_figures.py — Publication-quality figures for TauGenNet.

Usage (GPU required, ~15–20 min per mode on A100):
    python scripts/paper_figures.py --mode combined
    python scripts/paper_figures.py --mode ptau217
    python scripts/paper_figures.py --mode atrophy

All figures saved to results/figures/<mode>/paper/
    fig_a_multi_subject.pdf   — 6 test subjects: Real | Generated | |Diff|
    fig_b_table1_ablation.pdf — Table I: MRI ablation grouped bar chart
    fig_c_table2_nrmse.pdf    — Table II: NRMSE by plasma bin × ROI
    fig_d_metrics_summary.pdf — Overall metrics for all 3 paper models

Run --mode all to produce figures for all three paper models sequentially.
"""

import argparse, os, sys, pickle
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_fn

from src.config import DEVICE, VOL_SHAPE
from src.dataset import build_dataloaders
from src import dataset_combined as _dc
from src.diffusion import DiffusionSchedule
from src.inference import load_models, synthesize_tau_pet, synthesize_no_mri

# ── ROI definitions ────────────────────────────────────────────────────────────
ROI_DEFS = {
    "Parahippocampal":   (0.40, 0.55, 0.35, 0.65, 0.30, 0.55),
    "Fusiform":          (0.45, 0.60, 0.30, 0.70, 0.25, 0.50),
    "Inferior Temporal": (0.30, 0.55, 0.25, 0.75, 0.20, 0.55),
    "Hippocampus":       (0.42, 0.52, 0.40, 0.60, 0.35, 0.50),
    "Post. Cingulate":   (0.40, 0.55, 0.38, 0.62, 0.50, 0.70),
    "Entorhinal":        (0.43, 0.55, 0.38, 0.62, 0.28, 0.45),
}
ROI_SHORT = ["Parahip.", "Fusiform", "Inf.Temp.", "Hippocampus", "Post.Cing.", "Entorhinal"]
PLASMA_BINS  = [(0, 2), (2, 4), (4, 6), (6, 8), (10, float("inf"))]
BIN_LABELS   = ["0–2", "2–4", "4–6", "6–8", "10+"]
H, W, D = VOL_SHAPE


def roi_mean(vol, roi):
    y0, y1, x0, x1, z0, z1 = roi
    return vol[int(y0*H):int(y1*H), int(x0*W):int(x1*W), int(z0*D):int(z1*D)].mean()


def get_ptau(cond_i, mode):
    if mode == "combined":
        return cond_i[86].item()
    return cond_i.item()


# ── Data collection ────────────────────────────────────────────────────────────

def collect_test_data(ae, unet, schedule, encode_cond, latent_std, test_loader, mode,
                       n_steps=500):
    """Run synthesis on all test subjects. Returns list of dicts."""
    records = []
    torch.manual_seed(0)
    for pet_b, mri_b, cond_b in tqdm(test_loader, desc="Synthesising"):
        for j in range(pet_b.shape[0]):
            real_np = pet_b[j].squeeze().numpy()
            gen_full = synthesize_tau_pet(
                mri_b[j:j+1], cond_b[j], ae, unet, schedule,
                encode_cond, latent_std, device=DEVICE, n_steps=n_steps, sampler="ddpm"
            ).squeeze().numpy()
            gen_only = synthesize_no_mri(
                mri_b[j:j+1], cond_b[j], ae, unet, schedule,
                encode_cond, latent_std, device=DEVICE, n_steps=n_steps
            ).squeeze().numpy()

            mse_val  = float(np.mean((gen_full - real_np) ** 2))
            psnr_val = float(10 * np.log10(1.0 / (mse_val + 1e-10)))
            rec = {
                "real": real_np,
                "gen_full": gen_full,
                "gen_only": gen_only,
                "mse":           mse_val,
                "psnr":          psnr_val,
                "mae":           float(np.mean(np.abs(gen_full - real_np))),
                "ssim":          float(ssim_fn(real_np, gen_full, data_range=1.0)),
                "real_vol_mean": float(real_np.mean()),
            }
            if mode in ("ptau217", "combined"):
                rec["ptau"] = get_ptau(cond_b[j], mode)
            for rname, roi in ROI_DEFS.items():
                rec[f"real_{rname}"]  = float(roi_mean(real_np,  roi))
                rec[f"full_{rname}"]  = float(roi_mean(gen_full, roi))
                rec[f"only_{rname}"]  = float(roi_mean(gen_only, roi))
                rec[f"roi_mae_{rname}"] = abs(rec[f"full_{rname}"] - rec[f"real_{rname}"])
            records.append(rec)
    return records


# ── Figure A: Multi-subject comparison ────────────────────────────────────────

def fig_a_multi_subject(records, mode, save_path, n_subjects=6):
    subjects = records[:n_subjects]
    fig, axes = plt.subplots(n_subjects, 3, figsize=(9, n_subjects * 2.2))

    for row, rec in enumerate(subjects):
        real_np = rec["real"]
        gen_np  = rec["gen_full"]
        mid     = real_np.shape[2] // 2       # axial (D axis)
        real_s  = real_np[:, :, mid]
        gen_s   = gen_np[:, :, mid]
        diff_s  = np.abs(real_s - gen_s)

        axes[row, 0].imshow(real_s, cmap="hot", vmin=0, vmax=1)
        axes[row, 1].imshow(gen_s,  cmap="hot", vmin=0, vmax=1)
        im = axes[row, 2].imshow(diff_s, cmap="coolwarm", vmin=0, vmax=0.3)
        axes[row, 0].set_ylabel(f"S{row+1}  SSIM={rec['ssim']:.3f}", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")

    axes[0, 0].set_title("Real tau PET",      fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Generated tau PET", fontsize=11, fontweight="bold")
    axes[0, 2].set_title("|Difference|",      fontsize=11, fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Abs. error")

    ssims = [r["ssim"] for r in subjects]
    fig.suptitle(
        f"Tau PET synthesis  —  {mode} (paper arch)  |  "
        f"mean SSIM = {np.mean(ssims):.3f} ± {np.std(ssims):.3f}",
        fontsize=11, y=1.01
    )
    plt.subplots_adjust(left=0.12, right=0.90, hspace=0.05, wspace=0.05)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved Fig A → {save_path}")


# ── Figure B: Table I — MRI ablation bar chart ────────────────────────────────

def fig_b_table1(records, mode, save_path):
    label_map = {
        "atrophy":  ("MRI + Atrophy",        "Atrophy only"),
        "ptau217":  ("MRI + Plasma",          "Plasma only"),
        "combined": ("MRI + Atrophy + Plasma","Atrophy + Plasma only"),
    }
    lbl_full, lbl_only = label_map[mode]

    real_means  = {r: np.mean([rec[f"real_{r}"] for rec in records]) for r in ROI_DEFS}
    full_means  = {r: np.mean([rec[f"full_{r}"] for rec in records]) for r in ROI_DEFS}
    only_means  = {r: np.mean([rec[f"only_{r}"] for rec in records]) for r in ROI_DEFS}

    mse_full = [(real_means[r] - full_means[r])**2 for r in ROI_DEFS]
    mse_only = [(real_means[r] - only_means[r])**2 for r in ROI_DEFS]

    x = np.arange(len(ROI_SHORT))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x - w/2, mse_only, w, label=lbl_only,  color="#C44E52", alpha=0.85)
    b2 = ax.bar(x + w/2, mse_full, w, label=lbl_full,   color="#4C72B0", alpha=0.85)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.03,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(ROI_SHORT, fontsize=9)
    ax.set_ylabel("Group-level MSE")
    ax.set_title(f"Table I — MRI Ablation: {mode} (paper arch)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved Fig B → {save_path}")


# ── Figure C: Table II — NRMSE by plasma bin × ROI ────────────────────────────

def fig_c_table2(records, mode, save_path):
    if mode not in ("ptau217", "combined"):
        print(f"Fig C skipped: mode '{mode}' has no ptau217 conditioning.")
        return

    bin_real = {lbl: {r: [] for r in ROI_DEFS} for lbl in BIN_LABELS}
    bin_gen  = {lbl: {r: [] for r in ROI_DEFS} for lbl in BIN_LABELS}
    bin_n    = {lbl: 0 for lbl in BIN_LABELS}

    for rec in records:
        v   = rec["ptau"]
        lbl = next((l for (lo, hi), l in zip(PLASMA_BINS, BIN_LABELS) if lo <= v < hi), None)
        if lbl is None:
            continue
        for r in ROI_DEFS:
            bin_real[lbl][r].append(rec[f"real_{r}"])
            bin_gen[lbl][r].append(rec[f"full_{r}"])
        bin_n[lbl] += 1

    print("Subjects per plasma bin:")
    for lbl in BIN_LABELS:
        print(f"  {lbl:>4s} pg/mL : {bin_n[lbl]} subjects")

    nrmse = np.full((len(BIN_LABELS), len(ROI_DEFS)), np.nan)
    for i, lbl in enumerate(BIN_LABELS):
        for j, r in enumerate(ROI_DEFS):
            rv, gv = bin_real[lbl][r], bin_gen[lbl][r]
            if rv:
                nrmse[i, j] = abs(np.mean(rv) - np.mean(gv)) / (np.mean(rv) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    mask    = np.isnan(nrmse)
    display = np.where(mask, 0, nrmse)
    vmax    = np.nanmax(nrmse) if not np.all(mask) else 1.0
    im      = ax.imshow(display, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

    for i in range(len(BIN_LABELS)):
        for j in range(len(ROI_DEFS)):
            lbl_n = f"n={bin_n[BIN_LABELS[i]]}" if j == 0 else ""
            if mask[i, j]:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                ax.text(j, i, f"{nrmse[i, j]:.3f}", ha="center", va="center", fontsize=8)
            if j == 0 and bin_n[BIN_LABELS[i]] > 0:
                ax.text(-0.6, i, f"n={bin_n[BIN_LABELS[i]]}", ha="right", va="center",
                        fontsize=7, color="dimgray")

    ax.set_xticks(range(len(ROI_SHORT)))
    ax.set_xticklabels(ROI_SHORT, fontsize=9)
    ax.set_yticks(range(len(BIN_LABELS)))
    ax.set_yticklabels([f"{l} pg/mL" for l in BIN_LABELS], fontsize=9)
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Plasma p-tau217 bin")
    fig.colorbar(im, ax=ax, label="NRMSE")
    ax.set_title(f"Table II — NRMSE by plasma bin × region  ({mode}, paper arch)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved Fig C → {save_path}")


# ── Helper: grab first test sample as tensors ─────────────────────────────────

def grab_first_sample(test_loader):
    """Return (mri_tensor, cond_tensor) for the first test subject."""
    for pet_b, mri_b, cond_b in test_loader:
        return mri_b[0:1], cond_b[0]


# ── Helper: synthesize at a fixed plasma value ─────────────────────────────────

def synthesize_at_plasma(mri_t, base_cond, plasma_val, ae, unet, schedule,
                          encode_cond, latent_std, mode, n_steps=50):
    """Generate tau PET with plasma overridden to plasma_val (DDIM, fast)."""
    cond = base_cond.clone()
    if mode == "combined":
        cond[86] = plasma_val
    else:
        cond = torch.tensor([plasma_val], dtype=torch.float32)
    return synthesize_tau_pet(
        mri_t, cond, ae, unet, schedule, encode_cond, latent_std,
        device=DEVICE, n_steps=n_steps, sampler="ddim"
    ).squeeze().numpy()


# ── Figure 3: Training loss curve ─────────────────────────────────────────────

def fig_3_training_curves(diff_losses, mode, save_path):
    """Reproduces paper Fig. 3 (train curve only; val loss not saved in ckpt)."""
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


# ── Figure 4: Plasma sweep (paper Fig. 4) ────────────────────────────────────

PLASMA_SWEEP_VALS = [0.65, 3.65, 6.65, 10.65]

def fig_4_plasma_sweep(mri_t, base_cond, ae, unet, schedule, encode_cond,
                        latent_std, mode, save_path):
    """
    Reproduces paper Fig. 4: one subject's MRI fixed, tau PET generated at
    4 plasma values shown in axial / sagittal / coronal views.
    (Surface row from paper requires FreeSurfer — not reproduced here.)
    """
    torch.manual_seed(0)
    vols = [synthesize_at_plasma(mri_t, base_cond, pv, ae, unet, schedule,
                                  encode_cond, latent_std, mode)
            for pv in PLASMA_SWEEP_VALS]

    view_labels = ["Axial view", "Sagittal view", "Coronal view"]
    n_views, n_plasma = 3, len(PLASMA_SWEEP_VALS)
    fig, axes = plt.subplots(n_views, n_plasma, figsize=(n_plasma * 2.8, n_views * 2.5))

    for col, (vol, pv) in enumerate(zip(vols, PLASMA_SWEEP_VALS)):
        h, w, d = vol.shape
        slices = [vol[:, :, d // 2],   # axial
                  vol[:, w // 2, :],   # sagittal
                  vol[h // 2, :, :]]   # coronal
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


# ── Figure 5: Box plots generated vs. real by plasma bin (paper Fig. 5) ───────

def fig_5_boxplots(records, test_loader, ae, unet, schedule, encode_cond,
                    latent_std, mode, save_path):
    """
    Reproduces paper Fig. 5: box plots of generated (at fixed plasma) vs. real
    tau PET ROI averages, grouped by plasma bin. One subplot per ROI.
    """
    # Real distributions from existing records
    real_bins = {lbl: {r: [] for r in ROI_DEFS} for lbl in BIN_LABELS}
    for rec in records:
        if "ptau" not in rec:
            continue
        v   = rec["ptau"]
        lbl = next((l for (lo, hi), l in zip(PLASMA_BINS, BIN_LABELS) if lo <= v < hi), None)
        if lbl is None:
            continue
        for r in ROI_DEFS:
            real_bins[lbl][r].append(rec[f"real_{r}"])

    # Generated distributions: all test subjects at 4 fixed plasma values
    gen_bins = {pv: {r: [] for r in ROI_DEFS} for pv in PLASMA_SWEEP_VALS}
    torch.manual_seed(0)
    for pet_b, mri_b, cond_b in tqdm(test_loader, desc="Plasma sweep for Fig 5"):
        for j in range(mri_b.shape[0]):
            for pv in PLASMA_SWEEP_VALS:
                vol = synthesize_at_plasma(
                    mri_b[j:j+1], cond_b[j], pv,
                    ae, unet, schedule, encode_cond, latent_std, mode
                )
                for r, roi in ROI_DEFS.items():
                    gen_bins[pv][r].append(float(roi_mean(vol, roi)))

    # Build alternating box plot data per ROI
    roi_list  = list(ROI_DEFS.keys())
    real_lbls = BIN_LABELS
    gen_lbls  = [f"Gen {pv}" for pv in PLASMA_SWEEP_VALS]
    # interleave: real0, gen0, real1, gen1, ...
    x_labels = []
    for rl, gl in zip(real_lbls, gen_lbls):
        x_labels += [rl, gl]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    real_color = "#E8A0A0"
    gen_color  = "#A0C4E8"

    for ax, rname in zip(axes, roi_list):
        box_data   = []
        box_colors = []
        for rl, pv in zip(real_lbls, PLASMA_SWEEP_VALS):
            box_data.append(real_bins[rl][rname])
            box_colors.append(real_color)
            box_data.append(gen_bins[pv][rname])
            box_colors.append(gen_color)

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.2))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        # Red dashed mean lines
        for i, data in enumerate(box_data):
            if data:
                ax.hlines(np.mean(data), i + 0.7, i + 1.3,
                          colors="red", linestyles="dashed", linewidth=1.0)

        ax.set_xticks(range(1, len(x_labels) + 1))
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Avg. PET value in region", fontsize=8)
        ax.set_title(rname, fontsize=10, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=real_color, label="Real"),
                       Patch(facecolor=gen_color,  label="Generated")]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)
    fig.suptitle(
        f"PET Value Distributions by Plasma Bin and Source — {mode}",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved Fig 5 → {save_path}")


# ── Figure: ROI MSE boxplots across all 6 model runs ─────────────────────────

def fig_roi_mse_boxplots(all_records, save_path):
    """
    Boxplot of per-subject ROI-level MSE for all 6 model runs:
      atrophy+MRI, atrophy-only, ptau217+MRI, ptau217-only, combined+MRI, combined-only
    One subplot per ROI (2×3 grid).
    """
    run_labels = [
        "Atrophy\n+MRI",  "Atrophy\nonly",
        "Plasma\n+MRI",   "Plasma\nonly",
        "Both\n+MRI",     "Both\nonly",
    ]
    colors = [
        "#4C72B0", "#A8C4E8",   # atrophy: dark/light blue
        "#DD8452", "#F5C49A",   # ptau217: dark/light orange
        "#55A868", "#A8D4B4",   # combined: dark/light green
    ]

    roi_list = list(ROI_DEFS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, rname in zip(axes, roi_list):
        box_data = []
        for mode, full_key, only_key in [
            ("atrophy",  "full", "only"),
            ("atrophy",  "only", None),
            ("ptau217",  "full", "only"),
            ("ptau217",  "only", None),
            ("combined", "full", "only"),
            ("combined", "only", None),
        ]:
            recs = all_records.get(mode, [])
            if only_key is None:
                # "only" model — use gen_only vs real
                box_data.append([(rec[f"real_{rname}"] - rec[f"only_{rname}"]) ** 2
                                  for rec in recs])
            else:
                # "full" model — use gen_full vs real
                box_data.append([(rec[f"real_{rname}"] - rec[f"full_{rname}"]) ** 2
                                  for rec in recs])

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        ax.set_xticks(range(1, len(run_labels) + 1))
        ax.set_xticklabels(run_labels, fontsize=8)
        ax.set_ylabel("MSE  (roi_real − roi_gen)²", fontsize=8)
        ax.set_title(rname, fontsize=10, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Atrophy + MRI"),
        Patch(facecolor="#A8C4E8", label="Atrophy only"),
        Patch(facecolor="#DD8452", label="Plasma + MRI"),
        Patch(facecolor="#F5C49A", label="Plasma only"),
        Patch(facecolor="#55A868", label="Both + MRI"),
        Patch(facecolor="#A8D4B4", label="Both only"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("Per-subject ROI MSE across all 6 model runs", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved ROI MSE boxplots → {save_path}")


# ── Figure: Per-subject metric distributions (SSIM / PSNR / MAE) ──────────────

def fig_metrics_boxplots(all_records, save_path):
    """
    Side-by-side box plots of per-subject SSIM, PSNR, and MAE for all 3 modes.
    One column per metric, one box per mode.
    """
    modes  = ["atrophy", "ptau217", "combined"]
    labels = ["Atrophy + MRI", "Plasma + MRI", "Both + MRI"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    metrics = [
        ("ssim", "SSIM",    None),
        ("psnr", "PSNR (dB)", None),
        ("mae",  "MAE",     None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (key, ylabel, _) in zip(axes, metrics):
        data = [all_records.get(m, []) for m in modes]
        vals = [[r[key] for r in recs] for recs in data]
        # filter out empty lists so boxplot doesn't crash
        valid_vals   = [v for v in vals if v]
        valid_labels = [l for v, l in zip(vals, labels) if v]
        valid_colors = [c for v, c in zip(vals, colors) if v]

        bp = ax.boxplot(valid_vals, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], valid_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(valid_labels) + 1))
        ax.set_xticklabels(valid_labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")

    fig.suptitle("Per-subject metric distributions across model variants", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved metrics boxplots → {save_path}")


# ── Figure: Regional SUVR MAE across all 3 models ─────────────────────────────

def fig_roi_suvr_mae(all_records, save_path):
    """
    For each ROI: mean ± std of per-subject |SUVR_gen − SUVR_real|, grouped by model.
    2×3 subplot grid (one subplot per ROI), 3 bars per subplot (one per mode).
    """
    modes  = ["atrophy", "ptau217", "combined"]
    labels = ["Atrophy\n+MRI", "Plasma\n+MRI", "Both\n+MRI"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    roi_list = list(ROI_DEFS.keys())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    x = np.arange(len(modes))
    w = 0.55

    for ax, rname in zip(axes, roi_list):
        means, stds = [], []
        for m in modes:
            recs = all_records.get(m, [])
            vals = [r[f"roi_mae_{rname}"] for r in recs if f"roi_mae_{rname}" in r]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals)  if vals else 0)

        bars = ax.bar(x, means, w, color=colors, alpha=0.85,
                      yerr=stds, capsize=4, error_kw=dict(linewidth=1.2))
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.001,
                    f"{m:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Mean |SUVR gen − real|", fontsize=8)
        ax.set_title(rname, fontsize=10, fontweight="bold")

    fig.suptitle("Regional SUVR MAE (|generated − real| per ROI)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved ROI SUVR MAE → {save_path}")


# ── Figure D: Overall metrics comparison across all 3 paper models ─────────────
# Uses pre-computed values from eval_paper_3252131.out — no inference needed.

def fig_d_metrics_summary(save_path):
    # Global metrics from eval_paper_3252131.out
    models = ["atrophy\n(paper)", "ptau217\n(paper)", "combined\n(paper)"]
    maes   = [0.0534, 0.0576, 0.0767]
    mses   = [0.0059, 0.0093, 0.0083]
    psnrs  = [22.29,  20.31,  20.79]
    ssims  = [0.4478, 0.5451, 0.2197]
    # Per-subject mean ± std
    ssim_mean = [0.3855, 0.4928, 0.3163]
    ssim_std  = [0.1333, 0.0914, 0.1110]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(models))

    for ax, vals, ylabel, title in zip(
        axes,
        [maes, mses, psnrs, ssims],
        ["MAE", "MSE", "PSNR (dB)", "Global SSIM"],
        ["Mean Absolute Error", "Mean Squared Error", "PSNR", "SSIM (global)"],
    ):
        bars = ax.bar(x, vals, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{v:.4f}" if ylabel != "PSNR (dB)" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    # Overlay per-subject SSIM error bars on the SSIM panel
    ax_ssim = axes[3]
    ax_ssim.errorbar(x, ssim_mean, yerr=ssim_std, fmt="none",
                     color="black", capsize=4, linewidth=1.5, label="per-subject mean ± std")
    ax_ssim.legend(fontsize=7)

    fig.suptitle("Paper-arch model metrics  (from eval_paper_3252131.out)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved Fig D → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_mode(mode, n_steps=500):
    ckpt_dir = f"results/checkpoints/{mode}_paper"
    ckpt     = os.path.join(ckpt_dir, f"diff_{mode}.pt")
    fig_dir  = f"results/figures/{mode}/paper"
    os.makedirs(fig_dir, exist_ok=True)

    # Dataset
    if mode == "combined":
        train_ds, val_ds, test_ds, _, _, test_loader = _dc.build_dataloaders(mode=mode)
    else:
        train_ds, val_ds, test_ds, _, _, test_loader = build_dataloaders(mode=mode)

    print(f"\n{'='*60}")
    print(f"Mode: {mode}  |  test set: {len(test_ds)} subjects  |  ckpt: {ckpt}")
    print(f"{'='*60}")

    ae, unet, conditioner, latent_std, diff_losses = load_models(ckpt, mode)
    encode_cond = conditioner.encode
    schedule    = DiffusionSchedule()

    records = collect_test_data(ae, unet, schedule, encode_cond, latent_std,
                                 test_loader, mode, n_steps=n_steps)

    ssims = [r["ssim"] for r in records]
    mses  = [r["mse"]  for r in records]
    psnrs = [r["psnr"] for r in records]
    maes  = [r["mae"]  for r in records]
    print(f"Mean SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print(f"Mean MSE:  {np.mean(mses):.6f} ± {np.std(mses):.6f}")
    print(f"Mean PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
    print(f"Mean MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")

    # Save scalar records to cache (notebook loads from here)
    cache_dir  = "results/records"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{mode}_paper.pkl")
    scalar_records = [{k: v for k, v in r.items() if k not in ("real", "gen_full", "gen_only")}
                      for r in records]
    with open(cache_path, "wb") as f:
        pickle.dump(scalar_records, f)
    print(f"Records cached → {cache_path}")

    fig_a_multi_subject(records, mode, os.path.join(fig_dir, "fig_a_multi_subject.pdf"))
    fig_b_table1(records, mode,        os.path.join(fig_dir, "fig_b_table1_ablation.pdf"))
    fig_c_table2(records, mode,        os.path.join(fig_dir, "fig_c_table2_nrmse.pdf"))

    fig_3_training_curves(diff_losses, mode, os.path.join(fig_dir, "fig3_training_curve.pdf"))

    if mode in ("ptau217", "combined"):
        first_mri, first_cond = grab_first_sample(test_loader)
        fig_4_plasma_sweep(first_mri, first_cond, ae, unet, schedule, encode_cond,
                           latent_std, mode, os.path.join(fig_dir, "fig4_plasma_sweep.pdf"))
        fig_5_boxplots(records, test_loader, ae, unet, schedule, encode_cond,
                       latent_std, mode, os.path.join(fig_dir, "fig5_boxplots.pdf"))

    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["atrophy", "ptau217", "combined", "all"],
                   default="combined")
    p.add_argument("--n-steps", type=int, default=500)
    args = p.parse_args()

    modes = ["atrophy", "ptau217", "combined"] if args.mode == "all" else [args.mode]

    all_records = {}
    for mode in modes:
        all_records[mode] = run_mode(mode, n_steps=args.n_steps)

    # Fig D is always produced (uses pre-computed values, no inference)
    fig_d_metrics_summary("results/figures/paper_metrics_summary.pdf")

    # Multi-mode figures — require records from at least one mode
    if all_records:
        fig_metrics_boxplots(all_records, "results/figures/metrics_boxplots.pdf")
        fig_roi_suvr_mae(all_records,     "results/figures/roi_suvr_mae.pdf")

    # ROI MSE boxplots — requires records from all modes
    if set(modes) == {"atrophy", "ptau217", "combined"}:
        fig_roi_mse_boxplots(all_records, "results/figures/roi_mse_boxplots.pdf")

    print("\nAll figures complete.")
