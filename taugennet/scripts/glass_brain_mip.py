"""
Custom MIP-based glass brain visualization for TauGenNet.

Replaces nilearn rendering with direct numpy maximum intensity projections —
no temp-file roundtrip, full matplotlib control, publication-quality output.

Produces per mode:
  {mode}_figure_A_mip.png  — population mean MIP glass brain + metrics
  {mode}_figure_B_mip.png  — best / median / worst MIP glass brain + metrics

Usage:
    python scripts/glass_brain_mip.py --mode atrophy
    python scripts/glass_brain_mip.py --mode ptau217
    python scripts/glass_brain_mip.py --mode combined
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from skimage.metrics import structural_similarity as ssim_fn
from scipy.stats import pearsonr
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import VOL_SHAPE, FIGURES_DIR
from src.dataset_final import unnormalize

# ── design tokens ─────────────────────────────────────────────────────────────
BG   = "#0d0d0d"
FG   = "#e8e8e8"
DIM  = "#777777"
DPI  = 150

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "text.color":        FG,
    "axes.labelcolor":   FG,
    "axes.edgecolor":    "#2a2a2a",
    "font.family":       "sans-serif",
    "font.size":         10,
    "savefig.facecolor": BG,
    "savefig.edgecolor": "none",
})


# ── metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(real, gen):
    diff  = gen - real
    mse   = float(np.mean(diff ** 2))
    mae   = float(np.mean(np.abs(diff)))
    nrmse = float(np.sqrt(mse) / (real.max() - real.min() + 1e-8))
    ssim  = float(ssim_fn(real, gen, data_range=1.0))
    r, _  = pearsonr(real.ravel(), gen.ravel())
    return dict(mse=mse, mae=mae, nrmse=nrmse, ssim=ssim, pearson=float(r))


def _metric_block(m):
    return (f"NRMSE {m['nrmse']:.3f}\n"
            f"MAE   {m['mae']:.4f}\n"
            f"MSE   {m['mse']:.4f}\n"
            f"r     {m['pearson']:.3f}\n"
            f"SSIM  {m['ssim']:.3f}")


# ── data loading ──────────────────────────────────────────────────────────────

def _raw_norms(pet_path):
    vol = nib.load(pet_path).get_fdata().astype(np.float32)
    return float(vol.min()), float(vol.max())


def _load_test_data(mode, arch):
    if mode in ("atrophy", "ptau217"):
        from src.dataset_final import build_dataloaders
        _, _, test_ds, _, _, _ = build_dataloaders(mode=mode, use_dk_mask=True)
    else:
        from src.dataset_combined import build_dataloaders
        _, _, test_ds, _, _, _ = build_dataloaders(mode="combined", use_dk_mask=True)

    gen_dir = os.path.join(ROOT, "results", "generated", arch, mode)
    real_vols, gen_vols, real_norm, gen_norm = [], [], [], []

    for i in tqdm(range(len(test_ds)), desc=f"Loading {mode}"):
        pet, _, _ = test_ds[i]
        real_np = pet.squeeze(0).numpy()
        gen_path = os.path.join(gen_dir, f"subject_{i:03d}.npy")
        if not os.path.exists(gen_path):
            print(f"  WARNING: missing {gen_path}, skipping")
            continue
        gen_np = np.load(gen_path) * (real_np > 0)

        if hasattr(test_ds, "get_pet_norms"):
            pet_min, pet_max = test_ds.get_pet_norms(i)
        else:
            pet_min, pet_max = _raw_norms(test_ds.pet_paths[i])

        real_norm.append(real_np)
        gen_norm.append(gen_np)
        real_vols.append(unnormalize(real_np, pet_min, pet_max))
        gen_vols.append(unnormalize(gen_np, pet_min, pet_max))

    return real_vols, gen_vols, real_norm, gen_norm, test_ds


# ── MIP helpers ───────────────────────────────────────────────────────────────

def _mip(vol):
    """
    Maximum intensity projection along each axis.
    vol shape: (X=96, Y=112, Z=96)

    Returns three 2-D arrays oriented for display:
      axial    — project along Z, show XY plane (top-down view)
      coronal  — project along Y, show XZ plane (front view)
      sagittal — project along X, show YZ plane (side view)
    """
    axial    = np.rot90(np.max(vol, axis=2), k=1)
    coronal  = np.rot90(np.max(vol, axis=1), k=1)
    sagittal = np.rot90(np.max(vol, axis=0), k=1)
    return axial, coronal, sagittal


def _brain_outlines(brain_mask):
    """
    Compute MIP of the binary brain mask along each axis, then return as 2-D
    arrays in the same orientation as _mip().  Used to draw the brain silhouette
    contour on top of the signal projection.
    """
    axial    = np.rot90((np.max(brain_mask, axis=2) > 0).astype(float), k=1)
    coronal  = np.rot90((np.max(brain_mask, axis=1) > 0).astype(float), k=1)
    sagittal = np.rot90((np.max(brain_mask, axis=0) > 0).astype(float), k=1)
    return axial, coronal, sagittal


def _safe_vmax(arr, pct=99.5):
    return float(np.percentile(arr, pct))


def _mip_panel(ax, proj, outline, cmap, vmin, vmax):
    """
    Draw one MIP projection into ax with a brain silhouette contour overlay.
    outline: 2-D binary array (same shape as proj) — brain boundary mask MIP.
    Returns the AxesImage for colorbar use.
    """
    im = ax.imshow(proj, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", interpolation="bilinear", aspect="equal")
    # Brain silhouette — thin white contour at the mask boundary
    ax.contour(outline, levels=[0.5], colors=["white"],
               linewidths=[0.7], alpha=0.5, origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def _add_colorbar(fig, im, ax, label=""):
    """Thin vertical colorbar attached to the right edge of ax."""
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.004, pos.y0, 0.012, pos.height])
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.yaxis.set_tick_params(color=DIM, labelcolor=DIM, labelsize=7)
    cb.outline.set_edgecolor("#2a2a2a")
    if label:
        cb.set_label(label, color=DIM, fontsize=7)


# ── Figure A: population MIP ──────────────────────────────────────────────────

def figure_a_mip(real_vols, gen_vols, real_norm, gen_norm, mode, output_dir):
    mean_real = np.mean(real_vols, axis=0)
    mean_gen  = np.mean(gen_vols,  axis=0)
    mean_err  = mean_gen - mean_real

    hot_vmax = max(_safe_vmax(mean_real), _safe_vmax(mean_gen))
    err_vmax = float(np.percentile(np.abs(mean_err), 99.5))

    # Brain mask: union of all subjects' brain supports
    brain_mask = np.mean(real_vols, axis=0) > 0
    outlines = _brain_outlines(brain_mask)   # (axial, coronal, sagittal) 2-D masks

    all_m = [_compute_metrics(r, g) for r, g in zip(real_norm, gen_norm)]
    keys  = ["nrmse", "mae", "mse", "pearson", "ssim"]
    lbls  = ["NRMSE", "MAE", "MSE", "r", "SSIM"]
    means = {k: np.mean([m[k] for m in all_m]) for k in keys}
    stds  = {k: np.std ([m[k] for m in all_m]) for k in keys}
    summary = "   ".join(f"{l} {means[k]:.3f}±{stds[k]:.3f}" for k, l in zip(keys, lbls))

    vol_specs = [
        (mean_real, "hot",    0,         hot_vmax, "Mean Real PET  (SUVR)"),
        (mean_gen,  "hot",    0,         hot_vmax, "Mean Generated  (SUVR)"),
        (mean_err,  "RdBu_r", -err_vmax, err_vmax, "Mean Error  Gen − Real"),
    ]
    proj_labels = ["Axial", "Coronal", "Sagittal"]
    n_proj = 3

    # 3 projections (rows) × 3 volumes (cols)
    fig = plt.figure(figsize=(5 * len(vol_specs), 4 * n_proj + 1.2))
    gs  = mgridspec.GridSpec(n_proj + 1, len(vol_specs), figure=fig,
                             height_ratios=[*([1] * n_proj), 0.18],
                             hspace=0.04, wspace=0.12,
                             left=0.05, right=0.92, top=0.92, bottom=0.03)

    for col, (vol, cmap, vmin, vmax, title) in enumerate(vol_specs):
        axial, coronal, sagittal = _mip(vol)
        projs = [axial, coronal, sagittal]
        last_im = None
        for row, (proj, outline, plbl) in enumerate(zip(projs, outlines, proj_labels)):
            ax = fig.add_subplot(gs[row, col])
            im = _mip_panel(ax, proj, outline, cmap, vmin, vmax)
            last_im = im
            if col == 0:
                ax.set_ylabel(plbl, fontsize=10, color=FG, labelpad=5)
            if row == 0:
                ax.set_title(title, fontsize=11, color=FG, pad=8, fontweight="semibold")
        ax_bot = fig.add_subplot(gs[n_proj - 1, col])
        _add_colorbar(fig, last_im, ax_bot, label="SUVR" if col < 2 else "Δ SUVR")

    ax_m = fig.add_subplot(gs[n_proj, :])
    ax_m.axis("off")
    ax_m.text(0.5, 0.5,
              f"Per-subject mean ± std  (n={len(real_norm)}, normalized space)   |   {summary}",
              ha="center", va="center", fontsize=9.5, color=DIM,
              transform=ax_m.transAxes)

    fig.suptitle(f"{mode.upper()}  —  Population MIP Glass Brain",
                 fontsize=14, color=FG, y=0.96)
    out = os.path.join(output_dir, f"{mode}_figure_A_mip.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure B: subject MIP ─────────────────────────────────────────────────────

def figure_b_mip(real_vols, gen_vols, real_norm, gen_norm, mode, output_dir):
    mse_vals   = [float(np.mean((r - g) ** 2)) for r, g in zip(real_vols, gen_vols)]
    sorted_idx = np.argsort(mse_vals)
    n = len(sorted_idx)
    subjects = {"Best":   int(sorted_idx[0]),
                "Median": int(sorted_idx[n // 2]),
                "Worst":  int(sorted_idx[-1])}

    proj_labels = ["Axial", "Coronal", "Sagittal"]
    n_proj      = 3
    col_titles  = ["Real PET  (SUVR)", "Generated  (SUVR)", "Error  Gen − Real"]

    # Each subject gets 3 volume columns + 1 narrow spacer (except last subject)
    spacer_w = 0.3
    col_widths = []
    for s_i in range(len(subjects)):
        col_widths.extend([1.0, 1.0, 1.0])
        if s_i < len(subjects) - 1:
            col_widths.append(spacer_w)
    # +1 for the metrics column on the far right
    col_widths.append(1.1)

    n_gs_cols = len(col_widths)
    fig = plt.figure(figsize=(3.8 * (3 * len(subjects)) / len(subjects) * len(subjects) + 2, 
                               3.5 * n_proj + 1.5))
    gs  = mgridspec.GridSpec(n_proj + 1, n_gs_cols, figure=fig,
                             width_ratios=col_widths,
                             height_ratios=[*([1] * n_proj), 0.15],
                             hspace=0.04, wspace=0.06,
                             left=0.04, right=0.97, top=0.91, bottom=0.03)

    def vol_col(s_i, v_i):
        return s_i * (3 + 1) + v_i   # 3 vol cols + 1 spacer per subject

    metrics_col = n_gs_cols - 1      # far-right column reserved for metrics

    for s_i, (label, idx) in enumerate(subjects.items()):
        r_s, g_s = real_vols[idx], gen_vols[idx]
        err = g_s - r_s
        hot_vmax = max(_safe_vmax(r_s), _safe_vmax(g_s))
        err_vmax = float(np.percentile(np.abs(err), 99.5))
        m = _compute_metrics(real_norm[idx], gen_norm[idx])

        # Brain outline from this subject's real volume
        outlines = _brain_outlines(r_s > 0)

        vol_specs = [
            (r_s,  "hot",    0,         hot_vmax),
            (g_s,  "hot",    0,         hot_vmax),
            (err,  "RdBu_r", -err_vmax, err_vmax),
        ]

        last_im = [None, None, None]
        for v_i, (vol, cmap, vmin, vmax) in enumerate(vol_specs):
            axial, coronal, sagittal = _mip(vol)
            projs = [axial, coronal, sagittal]
            gc = vol_col(s_i, v_i)

            for row, (proj, outline, plbl) in enumerate(zip(projs, outlines, proj_labels)):
                ax = fig.add_subplot(gs[row, gc])
                im = _mip_panel(ax, proj, outline, cmap, vmin, vmax)
                last_im[v_i] = im
                if s_i == 0 and v_i == 0:
                    ax.set_ylabel(plbl, fontsize=10, color=FG, labelpad=5)
                if row == 0:
                    ax.set_title(col_titles[v_i], fontsize=9, color=DIM, pad=5)

            ax_bot = fig.add_subplot(gs[n_proj - 1, gc])
            _add_colorbar(fig, last_im[v_i], ax_bot,
                          label="SUVR" if v_i < 2 else "Δ SUVR")

        # Subject label centred over middle volume column
        ax_lbl = fig.add_subplot(gs[0, vol_col(s_i, 1)])
        ax_lbl.set_title(label, fontsize=13, color=FG, fontweight="bold", pad=28)

        # Metrics stacked in right column, one block per subject
        ax_m = fig.add_subplot(gs[s_i * (n_proj // len(subjects)):
                                  (s_i + 1) * (n_proj // len(subjects)) + (1 if s_i == len(subjects)-1 else 0),
                                  metrics_col])
        ax_m.axis("off")
        ax_m.text(0.08, 0.95, label,
                  transform=ax_m.transAxes, fontsize=11,
                  fontweight="bold", color=FG, va="top")
        ax_m.text(0.08, 0.78, _metric_block(m),
                  transform=ax_m.transAxes, fontsize=8.5,
                  color=DIM, va="top", family="monospace", linespacing=1.9)

    fig.suptitle(f"{mode.upper()}  —  Subject MIP Glass Brain  (best / median / worst)",
                 fontsize=13, color=FG, y=0.96)
    out = os.path.join(output_dir, f"{mode}_figure_B_mip.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Custom MIP glass brain for TauGenNet.")
    parser.add_argument("--mode", choices=["atrophy", "ptau217", "combined"],
                        default="atrophy")
    parser.add_argument("--arch", default="silu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(FIGURES_DIR, args.arch, "glass_brain")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=== MIP glass brain: mode={args.mode}  arch={args.arch} ===")
    real_vols, gen_vols, real_norm, gen_norm, test_ds = _load_test_data(args.mode, args.arch)
    if not real_vols:
        print("No subjects loaded — exiting.")
        return

    print("Rendering Figure A …")
    figure_a_mip(real_vols, gen_vols, real_norm, gen_norm, args.mode, args.output_dir)

    print("Rendering Figure B …")
    figure_b_mip(real_vols, gen_vols, real_norm, gen_norm, args.mode, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
