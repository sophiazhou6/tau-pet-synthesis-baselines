#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
scatter_plot.py — Scatter plot of generated vs real tau PET, per tau-relevant ROI.

Each point is the mean SUVR within one of 6 tau-relevant bounding-box ROIs
(same ROI_DEFS used in evaluate_final.py) for one test subject.
6 ROIs × N_test subjects, each ROI in a distinct color.

With --average: 6 per-region means ± SD error bars instead of individual subjects.

Usage:
  python scripts/scatter_plot.py --mode atrophy
  python scripts/scatter_plot.py --mode combined --arch relu --average
  python scripts/scatter_plot.py --mode atrophy --n-steps 50   # quick smoke-test
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy.stats import pearsonr

from src.config import DEVICE, VOL_SHAPE, FIGURES_DIR
from src import dataset_combined as _dc
from src import dataset_final as _dv2
from src.dataset_final import unnormalize as _unnormalize
from src.diffusion import DiffusionSchedule
from src.inference import load_models, synthesize_tau_pet

# Same ROI bounding boxes as evaluate_final.py (fractional coords: y0,y1,x0,x1,z0,z1)
ROI_DEFS = {
    "Parahippocampal":   (0.40, 0.55, 0.35, 0.65, 0.30, 0.55),
    "Fusiform":          (0.45, 0.60, 0.30, 0.70, 0.25, 0.50),
    "Inferior Temporal": (0.30, 0.55, 0.25, 0.75, 0.20, 0.55),
    "Hippocampus":       (0.42, 0.52, 0.40, 0.60, 0.35, 0.50),
    "Post. Cingulate":   (0.40, 0.55, 0.38, 0.62, 0.50, 0.70),
    "Entorhinal":        (0.43, 0.55, 0.38, 0.62, 0.28, 0.45),
}

ROI_COLORS = {
    "Parahippocampal":   "#e377c2",
    "Fusiform":          "#17becf",
    "Inferior Temporal": "#ff7f0e",
    "Hippocampus":       "#d62728",
    "Post. Cingulate":   "#2ca02c",
    "Entorhinal":        "#1f77b4",
}

H, W, D = VOL_SHAPE


def _roi_crop(vol, roi):
    y0, y1, x0, x1, z0, z1 = roi
    return vol[int(y0*H):int(y1*H), int(x0*W):int(x1*W), int(z0*D):int(z1*D)]


def _load_pet_suvr(pet_path):
    """Load raw PET, interpolate to VOL_SHAPE, return (suvr_vol_np, pmin, pmax)."""
    raw = nib.load(pet_path).get_fdata().astype(np.float32)
    vol = torch.tensor(raw).unsqueeze(0).unsqueeze(0)
    vol = F.interpolate(vol, size=VOL_SHAPE, mode="trilinear", align_corners=False).squeeze()
    vol_np = vol.numpy()
    return vol_np, float(vol_np.min()), float(vol_np.max())


def collect_roi_records(test_ds, ae, unet, schedule, encode_cond, latent_std,
                        n_steps, store_vols=False):
    """Run inference and extract per-ROI mean SUVR for every test subject.

    Returns (df, real_vols, gen_vols) where real_vols/gen_vols are lists of
    (H,W,D) SUVR arrays (populated only when store_vols=True, else empty lists).
    """
    records, real_vols, gen_vols = [], [], []
    for i in tqdm(range(len(test_ds)), desc="Subjects"):
        _, mri, cond = test_ds[i]

        gen_norm = synthesize_tau_pet(
            mri.unsqueeze(0).to(DEVICE),
            cond.unsqueeze(0).to(DEVICE),
            ae, unet, schedule, encode_cond, latent_std,
            n_steps=n_steps, sampler="ddpm",
        ).squeeze().cpu().numpy()

        pet_suvr, pmin, pmax = _load_pet_suvr(test_ds.pet_paths[i])
        gen_suvr = _unnormalize(gen_norm, pmin, pmax)

        if store_vols:
            real_vols.append(pet_suvr)
            gen_vols.append(gen_suvr)

        for rname, roi in ROI_DEFS.items():
            real_crop = _roi_crop(pet_suvr, roi)
            gen_crop  = _roi_crop(gen_suvr,  roi)
            records.append({
                "roi":  rname,
                "real": float(real_crop.mean()),
                "gen":  float(gen_crop.mean()),
                "subj": i,
            })

    return pd.DataFrame(records), real_vols, gen_vols


def compute_crosssubject_r(real_vols, gen_vols):
    """Voxelwise cross-subject Pearson r map from lists of (H,W,D) SUVR arrays."""
    real_stack = np.stack(real_vols, axis=0).astype(np.float64)
    gen_stack  = np.stack(gen_vols,  axis=0).astype(np.float64)
    real_c = real_stack - real_stack.mean(axis=0)
    gen_c  = gen_stack  - gen_stack.mean(axis=0)
    cov    = (real_c * gen_c).mean(axis=0)
    r_map  = (cov / (real_c.std(axis=0) * gen_c.std(axis=0) + 1e-8)).astype(np.float32)
    brain_mask = (real_stack > 0).any(axis=0)
    print(f"Cross-subject voxelwise Pearson r: mean={r_map[brain_mask].mean():.4f} "
          f"(N={len(real_vols)} subjects)")
    return r_map, brain_mask


def plot_voxelwise(r_map, brain_mask, real_vols, mode, out_dir):
    """Brain slice map + per-ROI bar chart of cross-subject Pearson r."""
    display = np.where(brain_mask, r_map, np.nan)
    H, W, D = display.shape
    slices  = {"Axial": display[H//2, :, :],
               "Sagittal": display[:, W//2, :],
               "Coronal": display[:, :, D//2]}

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#eeeeee")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, sl) in zip(axes, slices.items()):
        im = ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=-1, vmax=1,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r (cross-subject)")
    fig.suptitle(f"Voxelwise Cross-Subject Pearson r — {mode}", fontsize=13)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    map_path = os.path.join(out_dir, f"crosssubject_r_map_{mode}.png")
    fig.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {map_path}")

    # Per-ROI bar chart
    roi_r  = {rname: float(_roi_crop(r_map, roi).mean()) for rname, roi in ROI_DEFS.items()}
    rois   = list(roi_r.keys())
    values = [roi_r[r] for r in rois]
    colors = [cmap((v + 1) / 2) for v in values]
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    bars = ax2.barh(rois, values, color=colors, edgecolor="white", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlim(-1, 1)
    ax2.set_xlabel("Pearson r (cross-subject voxelwise mean)", fontsize=10)
    ax2.set_title(f"Per-ROI Cross-Subject Pearson r — {mode}", fontsize=11)
    for bar, v in zip(bars, values):
        ax2.text(v + (0.03 if v >= 0 else -0.03), bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig2.tight_layout()
    roi_path = os.path.join(out_dir, f"crosssubject_roi_r_{mode}.png")
    fig2.savefig(roi_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {roi_path}")


def plot_scatter(df, mode, output_path, average=False):
    fig, ax = plt.subplots(figsize=(7, 6))

    if average:
        grp = df.groupby("roi").agg(
            real_mean=("real", "mean"), real_std=("real", "std"),
            gen_mean=("gen",  "mean"), gen_std=("gen",  "std"),
        ).reset_index()

        for _, row in grp.iterrows():
            color = ROI_COLORS[row["roi"]]
            ax.errorbar(
                row["real_mean"], row["gen_mean"],
                xerr=row["real_std"], yerr=row["gen_std"],
                fmt="o", ms=8, alpha=0.9, color=color, label=row["roi"],
                elinewidth=1.2, capsize=4, linewidth=0, zorder=3,
            )
        real_all = grp["real_mean"].values
        gen_all  = grp["gen_mean"].values
    else:
        for rname, roi_df in df.groupby("roi"):
            color = ROI_COLORS[rname]
            ax.scatter(roi_df["real"], roi_df["gen"],
                       s=40, alpha=0.7, color=color, label=rname,
                       edgecolors="white", linewidths=0.3, zorder=3)
        real_all = df["real"].values
        gen_all  = df["gen"].values

    # Identity line
    lo = min(real_all.min(), gen_all.min())
    hi = max(real_all.max(), gen_all.max())
    margin = (hi - lo) * 0.03
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=1, zorder=2)

    rho, _ = pearsonr(real_all, gen_all)
    ax.text(0.05, 0.95, f"Pearson r = {rho:.3f}",
            transform=ax.transAxes, va="top", fontsize=10)

    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("Real mean SUVR per ROI", fontsize=11)
    ax.set_ylabel("Generated mean SUVR per ROI", fontsize=11)
    n_subj = df["subj"].nunique()
    suffix = " (region averages ± SD)" if average else f" ({n_subj} subjects × {len(ROI_DEFS)} ROIs)"
    ax.set_title(f"Regional tau PET — {mode}{suffix}", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")
    print(f"Pearson r = {rho:.4f}  (n={len(real_all)} points)")


def main():
    p = argparse.ArgumentParser(description="TauGenNet ROI scatter plot")
    p.add_argument("--mode",       choices=["atrophy", "ptau217", "combined"], required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to diffusion checkpoint (default: results/checkpoints/{arch}/{mode}/diff_{mode}.pt)")
    p.add_argument("--arch",       choices=["silu", "relu"], default="silu")
    p.add_argument("--n-steps",    type=int, default=500)
    p.add_argument("--output",     type=str, default=None,
                   help="Output PNG path (default: results/figures/{arch}/{mode}/scatter/scatter_roi_{mode}.png)")
    p.add_argument("--average",    action="store_true",
                   help="Plot per-ROI averages ± SD instead of individual subject points")
    p.add_argument("--voxelwise", action="store_true",
                   help="Also compute and save cross-subject voxelwise Pearson r map + per-ROI bar chart")
    args = p.parse_args()

    ckpt = args.checkpoint or f"results/checkpoints/{args.arch}/{args.mode}/diff_{args.mode}.pt"
    out  = args.output or os.path.join(FIGURES_DIR, args.arch, args.mode, "scatter", f"scatter_roi_{args.mode}.png")

    print(f"Mode:       {args.mode}")
    print(f"Checkpoint: {ckpt}")
    print(f"Steps:      {args.n_steps}")

    if args.mode == "combined":
        _, _, test_ds, *_ = _dc.build_dataloaders(mode="combined", use_dk_mask=False)
    else:
        _, _, test_ds, *_ = _dv2.build_dataloaders(mode=args.mode, use_dk_mask=False)
    print(f"Test subjects: {len(test_ds)}")

    ae, unet, conditioner, latent_std, _ = load_models(ckpt, args.mode, arch=args.arch)
    schedule = DiffusionSchedule()

    df, real_vols, gen_vols = collect_roi_records(
        test_ds, ae, unet, schedule, conditioner.encode, latent_std,
        args.n_steps, store_vols=args.voxelwise)
    print(f"ROI records collected: {len(df)}")

    plot_scatter(df, args.mode, out, average=args.average)

    if args.voxelwise:
        vox_out_dir = os.path.join(FIGURES_DIR, args.arch, args.mode, "scatter")
        r_map, brain_mask = compute_crosssubject_r(real_vols, gen_vols)
        plot_voxelwise(r_map, brain_mask, real_vols, args.mode, vox_out_dir)


if __name__ == "__main__":
    main()
