#!/usr/bin/env python3
"""
evaluation_paper.py — Paper-faithful evaluation of TauGenNet.

Implements the evaluation methodology from the paper (eqs 16-19):

  R_{i,g,k}    — mean SUVR within brain region k for subject i in plasma bin g
  R̄_real_{g,k} — group mean of real regional SUVR  (eq. 16)
  R̄_gen_{g,k}  — group mean of generated regional SUVR  (eq. 17)
  NRMSE_{g,k}  — sqrt((1/Ng) Σ(R_real − R_gen)²) / R̄_real_{g,k}  (eq. 18)
  SSIM         — full-volume structural similarity per subject  (eq. 19)

Additionally computes MSE, MAE, and Pearson's R using the same G×K framework
(plasma intervals × brain regions) so all metrics share one consistent table.

Key methodological choices (matching the paper):
  - All regional-mean metrics use RAW (unmasked) generated volumes in SUVR space.
    pmin > 0 for tau PET, so re-masking with (real > 0) after unnormalization is
    a no-op — the paper's eq. 18 simply takes the plain crop mean.
  - SSIM is applied to normalized [0,1] volumes (per-subject), then averaged.
  - Plasma p-tau217 value for combined mode: last element of the 87-dim conditioning
    vector ([atrophy z-scores (86) | ptau217 (1)]).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

from src.config import DEVICE, VOL_SHAPE, FIGURES_DIR
from src import dataset_final as _dataset_v2
from src import dataset_combined as _dataset_combined
from src.dataset_final import unnormalize as _unnormalize
from src.diffusion import DiffusionSchedule
from src.inference import load_models, synthesize_tau_pet


# ── constants ──────────────────────────────────────────────────────────────────

# Plasma p-tau217 intervals (pg/mL).  Six bins; no gap.
PLASMA_BINS = [
    (0,  2),
    (2,  4),
    (4,  6),
    (6,  8),
    (8,  10),
    (10, float("inf")),
]
BIN_LABELS = ["0-2", "2-4", "4-6", "6-8", "8-10", "10+"]

# Approximate ROI bounding boxes as fractions of (H, W, D).
ROI_DEFS = {
    "Parahippocampal":   (0.40, 0.55, 0.35, 0.65, 0.30, 0.55),
    "Fusiform":          (0.45, 0.60, 0.30, 0.70, 0.25, 0.50),
    "Inferior Temporal": (0.30, 0.55, 0.25, 0.75, 0.20, 0.55),
    "Hippocampus":       (0.42, 0.52, 0.40, 0.60, 0.35, 0.50),
    "Post. Cingulate":   (0.40, 0.55, 0.38, 0.62, 0.50, 0.70),
    "Entorhinal":        (0.43, 0.55, 0.38, 0.62, 0.28, 0.45),
}

H, W, D = VOL_SHAPE


# ── helpers ────────────────────────────────────────────────────────────────────

def _plasma_value(cond):
    """Return the scalar plasma p-tau217 from a subject's conditioning tensor.

    ptau217 mode: cond is shape (1,) → return directly.
    combined mode: cond is shape (87,) = [atrophy z-scores (86) | ptau217 (1)].
    """
    t = cond
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    t = np.asarray(t, dtype=np.float64).ravel()
    if t.size == 1:
        return float(t[0])
    return float(t[-1])


def _roi_crop(vol, roi):
    """Return the bounding-box crop of vol for the given ROI fraction tuple."""
    y0, y1, x0, x1, z0, z1 = roi
    return vol[int(y0 * H):int(y1 * H),
               int(x0 * W):int(x1 * W),
               int(z0 * D):int(z1 * D)]


def roi_mean_suvr(vol, roi):
    """Plain mean SUVR within an ROI bounding-box crop (no masking, per paper eq. 18)."""
    return float(_roi_crop(vol, roi).mean())


def _safe_ssim(real_vol, gen_vol, data_range=1.0):
    """3-D SSIM with adaptive window size for volumes smaller than 7³."""
    win = 7
    min_dim = min(real_vol.shape)
    if min_dim < win:
        win = max(3, min_dim if min_dim % 2 == 1 else min_dim - 1)
    return float(ssim(real_vol, gen_vol, data_range=data_range, win_size=win,
                      channel_axis=None))


# ── inference ──────────────────────────────────────────────────────────────────

def generate_all(ae, unet, schedule, encode_cond, latent_std, test_ds,
                 cond_mode, n_steps=500):
    """Run DDPM inference on every test subject.

    Returns
    -------
    real_norm   : list[np.ndarray]  — real tau PET in [0, 1]
    gen_raw     : list[np.ndarray]  — generated tau PET in [0, 1], unmasked
    real_suvr   : list[np.ndarray]  — real tau PET in SUVR
    gen_suvr    : list[np.ndarray]  — generated tau PET in SUVR, unmasked
    plasma_vals : list[float] | None — plasma p-tau217 scalar per subject,
                  or None when cond_mode == 'atrophy'
    """
    real_norm, gen_raw = [], []
    real_suvr, gen_suvr = [], []
    plasma_vals = [] if cond_mode in ("ptau217", "combined") else None

    for i in tqdm(range(len(test_ds)), desc="Generating"):
        pet, mri, cond = test_ds[i]

        real = pet.squeeze().numpy()

        with torch.no_grad():
            gen = synthesize_tau_pet(
                mri.unsqueeze(0).to(DEVICE),
                cond.unsqueeze(0).to(DEVICE),
                ae, unet, schedule, encode_cond, latent_std,
                n_steps=n_steps,
                sampler="ddpm",
            ).squeeze().cpu().numpy()

        real_norm.append(real)
        gen_raw.append(gen)

        # Unnormalize to SUVR.
        if hasattr(test_ds, "get_pet_norms"):
            pmin, pmax = test_ds.get_pet_norms(i)
        else:
            import nibabel as _nib
            _raw = _nib.load(test_ds.pet_paths[i]).get_fdata().astype(np.float32)
            pmin, pmax = float(_raw.min()), float(_raw.max())

        real_suvr.append(_unnormalize(real, pmin, pmax))
        gen_suvr.append(_unnormalize(gen,  pmin, pmax))

        if plasma_vals is not None:
            plasma_vals.append(_plasma_value(cond))

    return real_norm, gen_raw, real_suvr, gen_suvr, plasma_vals


# ── per-region, plasma-stratified metrics (eqs 16-18) ─────────────────────────

def _regional_means(suvr_vols):
    """Compute mean SUVR per subject per ROI.

    Returns an (N, K) array where N = subjects, K = ROIs.
    """
    rois = list(ROI_DEFS.keys())
    out = np.zeros((len(suvr_vols), len(rois)), dtype=np.float64)
    for i, vol in enumerate(suvr_vols):
        for k, roi in enumerate(ROI_DEFS.values()):
            out[i, k] = roi_mean_suvr(vol, roi)
    return out


def _metric_cell(r, g, metric):
    """Compute one cell of the G×K table from paired 1-D arrays r and g."""
    if r.size == 0:
        return float("nan")
    diff = r - g
    if metric == "nrmse":
        denom = float(r.mean())
        rmse  = float(np.sqrt(np.mean(diff ** 2)))
        return round(rmse / denom, 6) if abs(denom) > 1e-8 else float("nan")
    if metric == "mse":
        return float(np.mean(diff ** 2))
    if metric == "mae":
        return float(np.mean(np.abs(diff)))
    if metric == "pearson":
        if r.size < 2:
            return float("nan")
        rho, _ = pearsonr(r, g)
        return float(rho)
    raise ValueError(f"Unknown metric: {metric}")


def compute_plasma_metric_table(Rreal, Rgen, plasma_vals, metric):
    """Build one G×K DataFrame for the requested metric.

    Parameters
    ----------
    Rreal, Rgen : (N, K) arrays — per-subject regional means (SUVR)
    plasma_vals : (N,) array   — plasma p-tau217 scalar per subject
    metric      : one of {'nrmse', 'mse', 'mae', 'pearson'}
    """
    rois   = list(ROI_DEFS.keys())
    plasma = np.asarray(plasma_vals, dtype=np.float64)
    table  = pd.DataFrame(index=BIN_LABELS, columns=rois, dtype=float)
    counts = {}

    for (lo, hi), label in zip(PLASMA_BINS, BIN_LABELS):
        mask = (plasma >= lo) & (plasma < hi)
        counts[label] = int(mask.sum())
        for k, rname in enumerate(rois):
            table.loc[label, rname] = _metric_cell(Rreal[mask, k], Rgen[mask, k], metric)

    return table, counts


# ── whole-brain metrics (eq. 19 + supplementary) ──────────────────────────────

def compute_wholebrain_metrics(real_norm, gen_raw):
    """Per-subject whole-brain metrics in normalized [0, 1] space.

    gen_raw is unmasked; we apply brain-support mask (real > 0) here so that
    background leakage does not inflate error terms.

    Returns dict of metric_name → list[float] across subjects.
    """
    results = {"ssim": [], "mse": [], "mae": [], "pearson": []}

    for real, gen_u in zip(real_norm, gen_raw):
        gen = gen_u * (real > 0)   # suppress background leakage

        results["ssim"].append(_safe_ssim(real, gen, data_range=1.0))
        diff = real - gen
        results["mse"].append(float(np.mean(diff ** 2)))
        results["mae"].append(float(np.mean(np.abs(diff))))
        rho, _ = pearsonr(real.ravel(), gen.ravel())
        results["pearson"].append(float(rho))

    return results


# ── printing & figures ─────────────────────────────────────────────────────────

def print_plasma_table(table, counts, metric, mode):
    """Print one G×K metric table with subject counts."""
    label = {
        "nrmse":   "NRMSE  [eq. 18, SUVR, regional-mean across-subject]",
        "mse":     "MSE    [SUVR, regional-mean across-subject]",
        "mae":     "MAE    [SUVR, regional-mean across-subject]",
        "pearson": "Pearson [SUVR, regional-mean across-subject, Ng≥2]",
    }[metric]
    print(f"\n{'─'*70}")
    print(f"Plasma × Region {label}  —  {mode} mode")
    print(f"Subjects per bin: {counts}")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


def print_wholebrain_summary(wb, mode):
    """Print mean ± std of per-subject whole-brain metrics."""
    print(f"\n{'─'*70}")
    print(f"Whole-brain metrics (per-subject mean ± std)  —  {mode} mode")
    print(f"  (brain-support masked generated volumes, normalized [0,1] space)")
    for key, vals in wb.items():
        a = np.asarray(vals)
        print(f"  {key.upper():<12} {a.mean():.4f} ± {a.std():.4f}  "
              f"(min {a.min():.4f}, max {a.max():.4f})")


def plot_metric_heatmap(table, metric, mode, save_path):
    """Annotated heatmap of a G×K metric table."""
    cmaps = {"nrmse": "YlOrRd", "mse": "YlOrRd", "mae": "YlOrRd", "pearson": "RdYlGn"}
    data = table.values.astype(float)
    fig, ax = plt.subplots(figsize=(max(9, len(table.columns) * 1.5),
                                    0.7 * len(table.index) + 1.8))
    im = ax.imshow(data, cmap=cmaps.get(metric, "viridis"), aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.3f}",
                    ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label=metric.upper())
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Plasma p-tau217 interval (pg/mL)")
    ax.set_title(
        f"Plasma × Region  {metric.upper()}  [paper eq. 18 framework, SUVR]  —  {mode} mode"
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_wholebrain_boxplot(wb, mode, save_path):
    """Four-panel boxplot of per-subject whole-brain metrics."""
    keys = list(wb.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    rng = np.random.default_rng(0)
    for ax, key in zip(axes, keys):
        vals = np.asarray(wb[key])
        ax.boxplot(vals, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#a8c8e8"))
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(1 + jitter, vals, color="steelblue", s=18, alpha=0.7, zorder=3)
        ax.set_title(key.upper())
        ax.set_xticks([])
        ax.set_ylabel(key)
        ax.axhline(vals.mean(), color="red", lw=1.2, ls="--", label=f"mean {vals.mean():.3f}")
        ax.legend(fontsize=7)
    fig.suptitle(
        f"Per-subject whole-brain metrics  [brain-masked, [0,1] space]  —  {mode} mode",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def save_tables(tables, counts_by_metric, mode, out_dir):
    """Write each G×K metric table to a CSV and a markdown file."""
    os.makedirs(out_dir, exist_ok=True)
    md_lines = [f"# Paper Evaluation Tables — {mode} mode\n"]
    for metric, table in tables.items():
        csv_path = os.path.join(out_dir, f"paper_eval_{metric}_{mode}.csv")
        table.to_csv(csv_path)
        print(f"Saved: {csv_path}")
        md_lines.append(f"\n## {metric.upper()}\n")
        md_lines.append(f"Subjects per bin: {counts_by_metric[metric]}\n\n")
        md_lines.append(table.to_markdown(floatfmt=".4f"))
        md_lines.append("\n")
    md_path = os.path.join(out_dir, f"paper_eval_{mode}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {md_path}")


# ── entrypoint ─────────────────────────────────────────────────────────────────

_DIR_TO_SUBPATH = {
    "atrophy":   os.path.join("atrophy",  "72split"),
    "ptau217":   os.path.join("ptau217",  "72split"),
    "atrophy_v2": os.path.join("atrophy", "80split"),
    "ptau217_v2": os.path.join("ptau217", "80split"),
    "combined":  "atrophy_ptau217",
}

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Paper-faithful evaluation: G×K NRMSE/MSE/MAE/Pearson + SSIM"
    )
    p.add_argument("--mode", choices=["atrophy", "ptau217", "combined"],
                   required=True)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--figures-dir", type=str, default=None,
                   help="Subdirectory under results/figures/ for output figures")
    p.add_argument("--records-dir", type=str, default="results/records",
                   help="Directory for CSV/markdown table output")
    p.add_argument("--arch", choices=["silu", "relu"], default="silu")
    p.add_argument("--use-mask", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply DK atlas mask to data (default: on; use --no-use-mask to disable)")
    p.add_argument("--n-steps", type=int, default=500,
                   help="DDPM sampling steps (paper uses 500)")
    args = p.parse_args()

    COND_MODE = args.mode

    # Checkpoint
    if args.checkpoint_dir:
        CKPT_PATH = os.path.join(args.checkpoint_dir, f"diff_{COND_MODE}.pt")
    else:
        CKPT_PATH = f"results/checkpoints/{COND_MODE}/diff_{COND_MODE}.pt"

    # Figures directory
    if args.figures_dir:
        fig_subpath = args.figures_dir
    elif args.checkpoint_dir:
        key = os.path.basename(args.checkpoint_dir.rstrip("/"))
        fig_subpath = _DIR_TO_SUBPATH.get(key, key)
    else:
        fig_subpath = _DIR_TO_SUBPATH.get(COND_MODE, COND_MODE)

    FIG_DIR = os.path.join(FIGURES_DIR, fig_subpath, "paper_eval")
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Mode:     {COND_MODE}")
    print(f"Figures → {FIG_DIR}")
    print(f"Records → {args.records_dir}")

    # Dataset
    if args.arch == "relu":
        from src import dataset_v2_relu as _dataset_v2_relu
        from src import dataset_combined_relu as _dataset_combined_relu
        if COND_MODE == "combined":
            _, _, test_ds, *_ = _dataset_combined_relu.build_dataloaders(
                mode=COND_MODE, use_dk_mask=args.use_mask)
        else:
            _, _, test_ds, *_ = _dataset_v2_relu.build_dataloaders(
                mode=COND_MODE, use_dk_mask=args.use_mask)
    else:
        if COND_MODE == "combined":
            _, _, test_ds, *_ = _dataset_combined.build_dataloaders(
                mode=COND_MODE, use_dk_mask=args.use_mask)
        else:
            _, _, test_ds, *_ = _dataset_final.build_dataloaders(
                mode=COND_MODE, use_dk_mask=args.use_mask)

    print(f"Test subjects: {len(test_ds)}")

    # Models
    ae, unet, conditioner, latent_std, diff_losses = load_models(
        CKPT_PATH, COND_MODE, arch=args.arch)
    encode_cond = conditioner.encode
    schedule    = DiffusionSchedule()

    # ── Inference ──────────────────────────────────────────────────────────────
    real_norm, gen_raw, real_suvr, gen_suvr, plasma_vals = generate_all(
        ae, unet, schedule, encode_cond, latent_std, test_ds,
        cond_mode=COND_MODE, n_steps=args.n_steps,
    )

    # ── Whole-brain metrics (SSIM eq. 19 + MSE/MAE/Pearson) ───────────────────
    wb = compute_wholebrain_metrics(real_norm, gen_raw)
    print_wholebrain_summary(wb, COND_MODE)
    plot_wholebrain_boxplot(
        wb, COND_MODE,
        save_path=os.path.join(FIG_DIR, f"wholebrain_boxplot_{COND_MODE}.png"),
    )

    # ── Plasma-stratified regional metrics (eqs 16-18) ────────────────────────
    if plasma_vals is None:
        print(
            "\nSkipping plasma × region tables: 'atrophy' mode has no plasma p-tau217 "
            "conditioning. Use --mode ptau217 or --mode combined."
        )
    else:
        plasma_arr = np.asarray(plasma_vals, dtype=np.float64)
        print(f"\nPlasma p-tau217 range in test set: "
              f"{plasma_arr.min():.2f} – {plasma_arr.max():.2f}")

        # Pre-compute regional means once (N×K arrays).
        print("Computing regional means in SUVR space …")
        Rreal = _regional_means(real_suvr)
        Rgen  = _regional_means(gen_suvr)

        tables        = {}
        counts_by_met = {}
        for metric in ("nrmse", "mse", "mae", "pearson"):
            tbl, counts = compute_plasma_metric_table(
                Rreal, Rgen, plasma_vals, metric)
            tables[metric]        = tbl
            counts_by_met[metric] = counts
            print_plasma_table(tbl, counts, metric, COND_MODE)
            plot_metric_heatmap(
                tbl, metric, COND_MODE,
                save_path=os.path.join(
                    FIG_DIR, f"plasma_{metric}_{COND_MODE}.png"),
            )

        save_tables(tables, counts_by_met, COND_MODE, args.records_dir)

    print("\nDone.")
