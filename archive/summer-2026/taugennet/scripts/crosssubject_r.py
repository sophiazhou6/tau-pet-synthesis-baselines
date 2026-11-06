#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""Compute voxelwise cross-subject Pearson r from cached generations.

Usage:
    python scripts/crosssubject_r.py --mode atrophy
    python scripts/crosssubject_r.py --mode atrophy --arch relu
    python scripts/crosssubject_r.py --mode ptau217 --generated-dir results/generated/silu/ptau217

Figures saved to results/figures/<arch>/<mode>/ by default.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VOL_SHAPE = (96, 112, 96)
H, W, D = VOL_SHAPE

ROI_DEFS = {
    "Parahippocampal":   (0.40, 0.55, 0.35, 0.65, 0.30, 0.55),
    "Fusiform":          (0.45, 0.60, 0.30, 0.70, 0.25, 0.50),
    "Inferior Temporal": (0.30, 0.55, 0.25, 0.75, 0.20, 0.55),
    "Hippocampus":       (0.42, 0.52, 0.40, 0.60, 0.35, 0.50),
    "Post. Cingulate":   (0.40, 0.55, 0.38, 0.62, 0.50, 0.70),
    "Entorhinal":        (0.43, 0.55, 0.38, 0.62, 0.28, 0.45),
}

import src.dataset_v2 as _dataset_v2
import src.dataset_combined as _dataset_combined


def load_cached(test_ds, gen_dir, cond_mode):
    real_norm, gen_masked = [], []
    n_missing = 0
    for i in tqdm(range(len(test_ds)), desc="Loading"):
        path = os.path.join(gen_dir, f"subject_{i:03d}.npy")
        if not os.path.exists(path):
            n_missing += 1
            continue
        pet, _, _ = test_ds[i]
        real = pet.squeeze().numpy()
        gen  = np.load(path).astype(np.float32)
        real_norm.append(real)
        gen_masked.append(gen * (real > 0))
    if n_missing:
        print(f"WARNING: {n_missing} subject(s) missing from cache")
    return real_norm, gen_masked


def compute_crosssubject_r(real_norm, gen_masked):
    real_stack = np.stack(real_norm,  axis=0).astype(np.float64)
    gen_stack  = np.stack(gen_masked, axis=0).astype(np.float64)

    real_c = real_stack - real_stack.mean(axis=0)
    gen_c  = gen_stack  - gen_stack.mean(axis=0)
    cov    = (real_c * gen_c).mean(axis=0)
    r_map  = (cov / (real_c.std(axis=0) * gen_c.std(axis=0) + 1e-8)).astype(np.float32)

    brain_mask = (real_stack > 0).any(axis=0)
    mean_r = float(r_map[brain_mask].mean())
    print(f"Cross-subject voxelwise Pearson r: mean={mean_r:.4f} "
          f"(brain-masked, N={len(real_norm)} subjects)")
    return r_map, mean_r


def _roi_crop(vol, roi):
    y0, y1, x0, x1, z0, z1 = roi
    return vol[int(y0*H):int(y1*H), int(x0*W):int(x1*W), int(z0*D):int(z1*D)]


def compute_roi_r(r_map):
    roi_r = {}
    for name, roi in ROI_DEFS.items():
        roi_r[name] = float(_roi_crop(r_map, roi).mean())
        print(f"  {name:<22s} r = {roi_r[name]:.4f}")
    return roi_r


def plot_r_map(r_map, real_norm, mode, save_path):
    brain_mask = np.stack(real_norm, axis=0).any(axis=0)
    display    = np.where(brain_mask, r_map, np.nan)

    slices = {
        "Axial":    display[H//2, :, :],
        "Sagittal": display[:, W//2, :],
        "Coronal":  display[:, :, D//2],
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#eeeeee")
    for ax, (title, sl) in zip(axes, slices.items()):
        im = ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=-1, vmax=1,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r (cross-subject)")
    fig.suptitle(f"Voxelwise Cross-Subject Pearson r — {mode}", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"r map → {save_path}")


def plot_roi_r(roi_r, mode, save_path):
    rois   = list(roi_r.keys())
    values = [roi_r[r] for r in rois]
    cmap   = plt.cm.RdYlGn
    colors = [cmap((v + 1) / 2) for v in values]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(rois, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Pearson r (cross-subject voxelwise mean)", fontsize=10)
    ax.set_title(f"Per-ROI Cross-Subject Pearson r — {mode}", fontsize=11)
    for bar, v in zip(bars, values):
        ax.text(v + (0.03 if v >= 0 else -0.03), bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROI r → {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",          required=True, choices=["atrophy", "ptau217", "combined"])
    p.add_argument("--arch",          default="silu", choices=["silu", "relu"])
    p.add_argument("--generated-dir", default=None)
    p.add_argument("--figures-dir",   default=None,
                   help="Override figure output dir (default: results/figures/<arch>/<mode>/)")
    p.add_argument("--save-map",      default=None, help="Save r_map as .npy to this path")
    args = p.parse_args()

    if args.arch == "relu":
        from src import dataset_v2_relu as _dv2r
        from src import dataset_combined_relu as _dcr
        if args.mode == "combined":
            _, _, test_ds, *_ = _dcr.build_dataloaders(mode=args.mode)
        else:
            _, _, test_ds, *_ = _dv2r.build_dataloaders(mode=args.mode)
    else:
        if args.mode == "combined":
            _, _, test_ds, *_ = _dataset_combined.build_dataloaders(mode=args.mode)
        else:
            _, _, test_ds, *_ = _dataset_final.build_dataloaders(mode=args.mode)

    gen_dir = args.generated_dir or os.path.join("results", "generated", args.arch, args.mode)
    print(f"Loading from: {gen_dir}  ({len(test_ds)} test subjects)")

    real_norm, gen_masked = load_cached(test_ds, gen_dir, args.mode)
    r_map, mean_r = compute_crosssubject_r(real_norm, gen_masked)

    fig_dir = args.figures_dir or os.path.join("results", "figures", args.arch, args.mode)
    plot_r_map(r_map, real_norm, args.mode,
               os.path.join(fig_dir, f"crosssubject_r_map_{args.mode}.png"))
    roi_r = compute_roi_r(r_map)
    plot_roi_r(roi_r, args.mode,
               os.path.join(fig_dir, f"crosssubject_roi_r_{args.mode}.png"))

    if args.save_map:
        np.save(args.save_map, r_map)
        print(f"r_map saved → {args.save_map}")


if __name__ == "__main__":
    main()
