#!/usr/bin/env python3
"""
glass_brain_final.py — Consolidated TauGenNet visualizer, aligned with evaluate_final.py.

Produces six figures per mode:
  {mode}_glass_A.png   — population mean MIP glass brain (3 projections × real/gen/error)
  {mode}_slice_A.png   — population mean orthogonal slices (3 views × real/gen/error)
  {mode}_scatter_A.png — population real-vs-generated voxel density scatter
  {mode}_glass_B.png   — best/median/worst/pop-mean subject MIP glass brain
  {mode}_slice_B.png   — best/median/worst/pop-mean axial midslice
  {mode}_scatter_B.png — best/median/worst/pop-mean voxel density scatter

All conventions match evaluate_final.py exactly:
  - TAU_CMAP (blue→pink) for PET; Oranges for absolute error
  - Ranking by SSIM: best=argmax, median=argsort[n//2], worst=argmin
  - Global shared color range: pet_vmax=percentile(real_brain,99),
    err_vmax=percentile(all_abs_errors,95)
  - SUVR space for display; normalized [0,1] space for metrics
  - Brain masking: gen * (real > 0); use_dk_mask=True default

Usage:
    python scripts/glass_brain_final.py --mode atrophy
    python scripts/glass_brain_final.py --all
    python scripts/glass_brain_final.py --all --arch relu
    python scripts/glass_brain_final.py --mode atrophy --no-mask
"""

import argparse
import os
import sys
import tempfile

from scipy.ndimage import zoom as nd_zoom

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from skimage.metrics import structural_similarity as ssim_fn
from scipy.stats import pearsonr
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import VOL_SHAPE, FIGURES_DIR
from src.dataset_final import unnormalize

# ── colormap — matches evaluate_final.py exactly ──────────────────────────────
TAU_CMAP = LinearSegmentedColormap.from_list(
    "tau_pet",
    ["#08306b", "#2171b5", "#9ecae1", "#fbb4b9", "#f768a1", "#ae017e"],
)

BG  = "#0d0d0d"
FG  = "#e8e8e8"
DIM = "#777777"
DPI = 150

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "text.color":        FG,
    "axes.labelcolor":   FG,
    "axes.edgecolor":    "#2a2a2a",
    "xtick.color":       DIM,
    "ytick.color":       DIM,
    "font.family":       "sans-serif",
    "font.size":         10,
    "savefig.facecolor": BG,
    "savefig.edgecolor": "none",
})


# ── metrics ───────────────────────────────────────────────────────────────────

def _safe_ssim(real, gen, data_range=1.0):
    win = min(7, min(real.shape))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return float("nan")
    return float(ssim_fn(real, gen, data_range=data_range, win_size=win))


def _ssim_brain(real, gen, data_range=1.0):
    """SSIM on the brain bounding-box crop (real > 0), excluding zero padding.

    SSIM is a windowed spatial metric, so it can't be restricted to a 1-D vector of
    in-brain voxels like the other metrics. Cropping to the brain bounding box drops
    most of the DK-mask zero background — which otherwise inflates SSIM — while
    preserving the 3-D structure the metric needs.
    """
    nz = np.argwhere(real > 0)
    if nz.size == 0:
        return float("nan")
    lo, hi = nz.min(0), nz.max(0) + 1
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return _safe_ssim(real[sl], gen[sl], data_range=data_range)


def _compute_metrics(real_norm, gen_norm):
    """Whole-brain metrics in normalized [0,1] space — identical to evaluate_final.py.

    Voxel-wise metrics (Pearson/MSE/MAE/NRMSE) are restricted to in-brain voxels
    (real > 0), matching the scatter figures. Including the matched (0,0) background
    otherwise inflates Pearson and deflates the error terms. SSIM is computed on the
    brain bounding-box crop (it is a windowed metric and can't use a 1-D mask).
    """
    mask   = real_norm > 0
    r_in   = real_norm[mask]
    g_in   = gen_norm[mask]
    diff   = g_in - r_in
    mse    = float(np.mean(diff ** 2))
    mae    = float(np.mean(np.abs(diff)))
    nrmse  = float(np.sqrt(mse) / (r_in.max() - r_in.min() + 1e-8))
    ssim_v = _ssim_brain(real_norm, gen_norm)
    if r_in.std() < 1e-8 or g_in.std() < 1e-8:
        rho = float("nan")
    else:
        rho, _ = pearsonr(r_in, g_in)
    return dict(ssim=ssim_v, pearson=float(rho), nrmse=nrmse, mse=mse, mae=mae)


def _mstr(ssim, pearson, nrmse, mse, mae):
    return (f"SSIM={ssim:.3f}  Pearson={pearson:.3f}  "
            f"NRMSE={nrmse:.3f}  MSE={mse:.4f}  MAE={mae:.4f}")


# ── global color range — matches evaluate_final.py compute_suvr_range ─────────

def compute_suvr_range(real_suvr, gen_suvr):
    all_brain = np.concatenate([v[v > 0].ravel() for v in real_suvr])
    pet_vmax  = float(np.percentile(all_brain, 99))
    all_err   = np.concatenate([np.abs(r - g).ravel() for r, g in zip(real_suvr, gen_suvr)])
    err_vmax  = float(np.percentile(all_err, 95))
    return 0.0, pet_vmax, 0.0, err_vmax


# ── SSIM ranking — identical to evaluate_final.py plot_subject_comparison ─────

def _rank_by_ssim(ssims):
    arr = np.array(ssims)
    n   = len(arr)
    return {
        "Best":   int(np.argmax(arr)),
        "Median": int(np.argsort(arr)[n // 2]),
        "Worst":  int(np.argmin(arr)),
    }


# Generic per-subject ranking (Best=argmax, Median=mid, Worst=argmin) for any metric key.
# 'ssim'/'pearson' are higher-is-better. Pass by='pearson' to rank glass/slice/scatter B
# figures by voxel Pearson instead of SSIM. _rank_by_ssim kept for backward compatibility.
_RANK_LABEL = {"ssim": "SSIM", "pearson": "Pearson", "nrmse": "NRMSE",
               "mse": "MSE", "mae": "MAE"}


# Error metrics where LOWER is better -> Best = argmin (others are higher-is-better).
_LOWER_BETTER = {"nrmse", "mse", "mae"}


def _rank_subjects(all_metrics, by="ssim"):
    arr   = np.array([m[by] for m in all_metrics])
    order = np.argsort(arr)                       # ascending
    n     = len(arr)
    lo, hi = int(order[0]), int(order[-1])
    best, worst = (lo, hi) if by in _LOWER_BETTER else (hi, lo)
    return {"Best": best, "Median": int(order[n // 2]), "Worst": worst}


# ── data loading ──────────────────────────────────────────────────────────────

def _raw_norms(pet_path):
    """Fallback SUVR norms for combined mode (no get_pet_norms)."""
    vol = nib.load(pet_path).get_fdata().astype(np.float32)
    return float(vol.min()), float(vol.max())


def _load_test_data(mode, arch, use_dk_mask=True):
    # Mirror evaluate_final.py's arch-aware dataset selection exactly. The relu
    # datasets use config_relu (VOL_SHAPE 160×160×96), so the cached .npy files
    # must be paired with the same dataset class that generated them — otherwise
    # the real PET is at the wrong resolution and SSIM ranking diverges.
    if arch == "relu":
        if mode == "combined":
            from src.dataset_combined_relu import build_dataloaders
            _, _, test_ds, _, _, _ = build_dataloaders(mode="combined", use_dk_mask=use_dk_mask)
        else:
            from src.dataset_v2_relu import build_dataloaders
            _, _, test_ds, _, _, _ = build_dataloaders(mode=mode, use_dk_mask=use_dk_mask)
    else:
        if mode == "combined":
            from src.dataset_combined import build_dataloaders
            _, _, test_ds, _, _, _ = build_dataloaders(mode="combined", use_dk_mask=use_dk_mask)
        else:
            # ptau217_mlp shares the ptau217 data (1-dim scalar conditioning)
            dataset_mode = "ptau217" if mode == "ptau217_mlp" else mode
            from src.dataset_final import build_dataloaders
            _, _, test_ds, _, _, _ = build_dataloaders(mode=dataset_mode, use_dk_mask=use_dk_mask)

    gen_dir = os.path.join(ROOT, "results", "generated", arch, mode)
    real_vols, gen_vols, real_norm_list, gen_norm_list = [], [], [], []

    for i in tqdm(range(len(test_ds)), desc=f"Loading {mode}"):
        pet, _, _ = test_ds[i]
        real_np   = pet.squeeze(0).numpy()

        gen_path = os.path.join(gen_dir, f"subject_{i:03d}.npy")
        if not os.path.exists(gen_path):
            print(f"  WARNING: missing {gen_path}, skipping")
            continue

        gen_np = np.load(gen_path)
        if gen_np.shape != real_np.shape:
            scale = tuple(t / s for t, s in zip(real_np.shape, gen_np.shape))
            gen_np = nd_zoom(gen_np, scale, order=1)
        gen_np_masked = gen_np * (real_np > 0)  # masked normalized — for metrics

        if hasattr(test_ds, "get_pet_norms"):
            pet_min, pet_max = test_ds.get_pet_norms(i)
        else:
            pet_min, pet_max = _raw_norms(test_ds.pet_paths[i])

        gen_suvr = unnormalize(gen_np, pet_min, pet_max)
        gen_suvr[real_np <= 0] = 0  # mask after unnormalize for display

        real_norm_list.append(real_np)
        gen_norm_list.append(gen_np_masked)
        real_vols.append(unnormalize(real_np, pet_min, pet_max))
        gen_vols.append(gen_suvr)

    return real_vols, gen_vols, real_norm_list, gen_norm_list, test_ds


# ── shared colorbar helper ────────────────────────────────────────────────────

def _add_colorbar(fig, ax, vmin, vmax, cmap, label, shrink=0.8):
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=shrink, pad=0.02, label=label)
    cb.set_label(label, color=DIM, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=DIM, labelcolor=DIM, labelsize=7)
    cb.outline.set_edgecolor("#2a2a2a")


def _add_sm_colorbar(fig, axes_list, cmap, vmin, vmax, label):
    """Colorbar alongside a list of axes (glass brain figures)."""
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes_list, shrink=0.6, pad=0.02, fraction=0.03)
    cb.set_label(label, color=DIM, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=DIM, labelcolor=DIM, labelsize=7)
    cb.outline.set_edgecolor("#2a2a2a")


# ── slice panel helpers (origin="lower" — matches evaluate_final.py) ──────────

def _imshow_pet(ax, slc, vmin, vmax):
    ax.imshow(slc, cmap=TAU_CMAP, vmin=vmin, vmax=vmax, origin="lower")
    ax.axis("off")


def _imshow_err(ax, slc, vmax):
    ax.imshow(slc, cmap="Oranges", vmin=0, vmax=vmax, origin="lower")
    ax.axis("off")


# ── nilearn glass brain helpers ───────────────────────────────────────────────

from nilearn import plotting as nlplot


def _compute_affine(test_ds, vol_shape):
    """Scale the first subject's NIfTI affine to match the resampled volume shape.

    vol_shape is the ACTUAL shape of the loaded volumes (relu uses config_relu's
    160×160×96, silu uses 96×112×96) — using the module-level VOL_SHAPE import here
    would mis-scale the affine for relu and push projections outside the glass-brain
    outline.
    """
    orig  = nib.load(test_ds.pet_paths[0])
    scale = np.array(orig.shape[:3], float) / np.array(vol_shape, float)
    A     = orig.affine.copy()
    for i in range(3):
        A[:3, i] *= scale[i]
    return A


def _to_nifti(vol, affine):
    return nib.Nifti1Image(vol.astype(np.float32), affine=affine)


def _render_glass(vol, affine, cmap, vmin, vmax, threshold=1e-3):
    """Render one glass brain panel → RGBA numpy array via temp PNG."""
    nii = _to_nifti(vol, affine)
    nl_fig = plt.figure(figsize=(8, 3.5), facecolor="black")
    display = nlplot.plot_glass_brain(
        nii,
        figure=nl_fig,
        display_mode="ortho",
        colorbar=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        black_bg=True,
        threshold=threshold,
        annotate=False,
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    display.savefig(tmp, dpi=DPI)
    display.close()
    plt.close(nl_fig)
    img = plt.imread(tmp)
    os.unlink(tmp)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A: POPULATION LEVEL
# ═══════════════════════════════════════════════════════════════════════════════

def figure_a_glass(real_vols, gen_vols, real_norm_list, gen_norm_list,
                   mode, output_dir, pet_vmin, pet_vmax, err_vmax, all_metrics,
                   affine):
    """Population mean glass brain via nilearn — real / generated / abs error."""
    real_arr  = np.stack(real_vols)
    gen_arr   = np.stack(gen_vols)
    mean_real = real_arr.mean(axis=0)
    mean_gen  = gen_arr.mean(axis=0)
    mean_err  = np.abs(real_arr - gen_arr).mean(axis=0)

    means = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    stds  = {k: np.std( [m[k] for m in all_metrics]) for k in all_metrics[0]}
    summary = (f"SSIM {means['ssim']:.3f}±{stds['ssim']:.3f}   "
               f"Pearson {means['pearson']:.3f}±{stds['pearson']:.3f}   "
               f"NRMSE {means['nrmse']:.3f}±{stds['nrmse']:.3f}   "
               f"MSE {means['mse']:.4f}±{stds['mse']:.4f}")

    col_specs = [
        (mean_real, TAU_CMAP,  pet_vmin, pet_vmax, 1e-3, "Mean Real PET (SUVR)"),
        (mean_gen,  TAU_CMAP,  pet_vmin, pet_vmax, 1e-3, "Mean Generated (SUVR)"),
        (mean_err,  "Oranges", 0,        err_vmax,  1e-3, "Mean Abs Error (SUVR)"),
    ]

    print("    Rendering nilearn panels for Figure A…")
    imgs = [_render_glass(vol, affine, cmap, vmin, vmax, thr)
            for vol, cmap, vmin, vmax, thr, _ in col_specs]

    fig, axes = plt.subplots(1, 3, figsize=(21, 9))
    for ax, img, (_, cmap, vmin, vmax, _, title) in zip(axes, imgs, col_specs):
        ax.imshow(img, aspect="equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11, color=FG, fontweight="semibold", pad=6)

    _add_colorbar(fig, axes[0], pet_vmin, pet_vmax, TAU_CMAP,  "SUVR",             shrink=0.7)
    _add_colorbar(fig, axes[1], pet_vmin, pet_vmax, TAU_CMAP,  "SUVR",             shrink=0.7)
    _add_colorbar(fig, axes[2], 0,        err_vmax,  "Oranges", "Abs Error (SUVR)", shrink=0.7)

    fig.text(0.5, 0.01,
             f"Per-subject mean ± std  (n={len(real_vols)}, normalized space)   |   {summary}",
             ha="center", fontsize=9, color=DIM)

    fig.suptitle(f"{mode.upper()}  —  Population Glass Brain", fontsize=14, color=FG, y=1.0)
    out = os.path.join(output_dir, "glass_A.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


def figure_a_slice(real_vols, gen_vols, mode, output_dir,
                   pet_vmin, pet_vmax, err_vmax, all_metrics):
    """Population mean orthogonal slices — matches evaluate_final.plot_population_comparison."""
    real_arr  = np.stack(real_vols)
    gen_arr   = np.stack(gen_vols)
    mean_real = real_arr.mean(axis=0)
    mean_gen  = gen_arr.mean(axis=0)
    mean_err  = np.abs(real_arr - gen_arr).mean(axis=0)

    # Same view order as evaluate_final.py
    views = [("Axial", 2), ("Sagittal", 0), ("Coronal", 1)]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for row_i, (view_name, axis_dim) in enumerate(views):
        mid = mean_real.shape[axis_dim] // 2
        _imshow_pet(axes[row_i, 0], np.take(mean_real, mid, axis=axis_dim), pet_vmin, pet_vmax)
        _imshow_pet(axes[row_i, 1], np.take(mean_gen,  mid, axis=axis_dim), pet_vmin, pet_vmax)
        _imshow_err(axes[row_i, 2], np.take(mean_err,  mid, axis=axis_dim), err_vmax)
        if row_i == 0:
            axes[0, 0].set_title("Mean Real PET (SUVR)",      fontsize=11,
                                 color=FG, fontweight="bold")
            axes[0, 1].set_title("Mean Generated (SUVR)",     fontsize=11,
                                 color=FG, fontweight="bold")
            axes[0, 2].set_title("Mean Abs Error (SUVR)",     fontsize=11,
                                 color=FG, fontweight="bold")
        axes[row_i, 0].set_ylabel(view_name, fontsize=10, color=FG)

    _add_colorbar(fig, axes[:, 1], pet_vmin, pet_vmax, TAU_CMAP, "SUVR")
    _add_colorbar(fig, axes[:, 2], 0, err_vmax, "Oranges", "Abs Error (SUVR)")

    means = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    stds  = {k: np.std( [m[k] for m in all_metrics]) for k in all_metrics[0]}
    fig.suptitle(
        f"{mode.upper()}  —  Population Slices  (n={len(real_vols)} test subjects)\n"
        f"SSIM {means['ssim']:.3f}±{stds['ssim']:.3f}   "
        f"Pearson {means['pearson']:.3f}±{stds['pearson']:.3f}   "
        f"NRMSE {means['nrmse']:.3f}±{stds['nrmse']:.3f}   "
        f"MSE {means['mse']:.4f}±{stds['mse']:.4f}",
        fontsize=11, color=FG)

    out = os.path.join(output_dir, "slice_A.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


def figure_a_scatter(real_vols, gen_vols, pet_vmax, mode, output_dir):
    """Population voxel-level real vs generated density scatter (all subjects pooled)."""
    all_real = np.concatenate([v[v > 0].ravel() for v in real_vols])
    all_gen  = np.concatenate([g[r > 0].ravel() for r, g in zip(real_vols, gen_vols)])
    rho, _   = pearsonr(all_real, all_gen)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    hb = ax.hexbin(all_real, all_gen, gridsize=100, cmap="hot",
                   extent=[0, pet_vmax, 0, pet_vmax],
                   norm=mcolors.LogNorm(vmin=1))
    ax.plot([0, pet_vmax], [0, pet_vmax], "w--", linewidth=1.0, alpha=0.5, label="y = x")
    ax.set_xlim(0, pet_vmax); ax.set_ylim(0, pet_vmax)
    ax.set_xlabel("Real SUVR", fontsize=10, color=FG)
    ax.set_ylabel("Generated SUVR", fontsize=10, color=FG)
    ax.text(0.05, 0.95, f"Pearson r = {rho:.3f}\nn = {len(all_real):,} voxels",
            transform=ax.transAxes, fontsize=9, color=FG, va="top", family="monospace")
    cb = fig.colorbar(hb, ax=ax, label="Voxel count (log)")
    cb.set_label("Voxel count (log)", color=DIM, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=DIM, labelcolor=DIM, labelsize=7)
    cb.outline.set_edgecolor("#2a2a2a")
    fig.suptitle(f"{mode.upper()}  —  Population Voxel Scatter  (all {len(real_vols)} subjects)",
                 fontsize=12, color=FG)
    plt.tight_layout()

    out = os.path.join(output_dir, "scatter_A.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE B: SUBJECT LEVEL (best / median / worst / population mean)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_rows(real_vols, gen_vols, real_arr, gen_arr, all_metrics, ranked):
    """Assemble the 4 subject rows used by figure_b_* functions."""
    mean_real = real_arr.mean(axis=0)
    mean_gen  = gen_arr.mean(axis=0)
    mean_err  = np.abs(real_arr - gen_arr).mean(axis=0)
    pop_mask  = (real_arr > 0).any(axis=0)

    B, M, W = ranked["Best"], ranked["Median"], ranked["Worst"]

    def row(idx, label):
        m = all_metrics[idx]
        ms = _mstr(m["ssim"], m["pearson"], m["nrmse"], m["mse"], m["mae"])
        return dict(label=label, idx=idx,
                    r=real_vols[idx], g=gen_vols[idx],
                    err=np.abs(real_vols[idx] - gen_vols[idx]),
                    mask=real_vols[idx] > 0, mstr=ms)

    return [
        row(B, "Best"),
        row(M, "Median"),
        row(W, "Worst"),
        dict(label="Pop. Mean", idx=None,
             r=mean_real, g=mean_gen, err=mean_err,
             mask=pop_mask, mstr=None),
    ]


def figure_b_glass(real_vols, gen_vols, real_norm_list, gen_norm_list,
                   mode, output_dir, pet_vmin, pet_vmax, err_vmax,
                   all_metrics, ranked, affine, rank_by="ssim"):
    """Best/median/worst/pop-mean glass brain via nilearn, global shared color range."""
    real_arr  = np.stack(real_vols)
    gen_arr   = np.stack(gen_vols)
    rows_spec = _build_rows(real_vols, gen_vols, real_arr, gen_arr, all_metrics, ranked)
    col_titles = ["Real PET (SUVR)", "Generated (SUVR)", "Abs Error (SUVR)"]
    n_rows = len(rows_spec)

    print(f"    Rendering nilearn panels for Figure B ({n_rows * 3} panels)…")
    grid_imgs = []
    for rd in rows_spec:
        grid_imgs.append([
            _render_glass(rd["r"],   affine, TAU_CMAP,  pet_vmin, pet_vmax, 1e-3),
            _render_glass(rd["g"],   affine, TAU_CMAP,  pet_vmin, pet_vmax, 1e-3),
            _render_glass(rd["err"], affine, "Oranges", 0,        err_vmax,  1e-3),
        ])

    fig, axes = plt.subplots(n_rows, 3, figsize=(21, 8 * n_rows),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.04})

    for r_i, (rd, row_imgs) in enumerate(zip(rows_spec, grid_imgs)):
        for c_i, img in enumerate(row_imgs):
            ax = axes[r_i, c_i]
            ax.imshow(img, aspect="equal")
            ax.axis("off")
            if r_i == 0:
                ax.set_title(col_titles[c_i], fontsize=11, color=FG,
                             fontweight="bold", pad=6)
        # Row label on leftmost panel
        axes[r_i, 0].set_ylabel(rd["label"], fontsize=11, color=FG,
                                fontweight="bold", labelpad=8)
        # Metrics below middle panel
        if rd["mstr"] is not None:
            axes[r_i, 1].set_xlabel(rd["mstr"], fontsize=7.5, color=DIM, ha="center")

    _add_colorbar(fig, axes[:, 1], pet_vmin, pet_vmax, TAU_CMAP,  "SUVR",             shrink=0.6)
    _add_colorbar(fig, axes[:, 2], 0,        err_vmax,  "Oranges", "Abs Error (SUVR)", shrink=0.6)

    fig.suptitle(f"{mode.upper()}  —  Subject Glass Brain  (ranked by {_RANK_LABEL.get(rank_by, rank_by)})",
                 fontsize=13, color=FG, y=1.01)
    out = os.path.join(output_dir, "glass_B.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


def figure_b_slice(real_vols, gen_vols, mode, output_dir,
                   pet_vmin, pet_vmax, err_vmax, all_metrics, ranked, rank_by="ssim"):
    """Axial midslice — matches evaluate_final.plot_subject_comparison exactly."""
    real_arr  = np.stack(real_vols)
    gen_arr   = np.stack(gen_vols)
    rows_spec = _build_rows(real_vols, gen_vols, real_arr, gen_arr, all_metrics, ranked)

    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    col_titles = ["Real PET (SUVR)", "Generated PET (SUVR)", "Abs Error (SUVR)"]

    for row_i, rd in enumerate(rows_spec):
        mid = rd["r"].shape[2] // 2   # axial midslice (z = 48), same as evaluate_final.py
        _imshow_pet(axes[row_i, 0], rd["r"][:, :, mid], pet_vmin, pet_vmax)
        _imshow_pet(axes[row_i, 1], rd["g"][:, :, mid], pet_vmin, pet_vmax)
        _imshow_err(axes[row_i, 2], rd["err"][:, :, mid], err_vmax)

        if row_i == 0:
            for ci, t in enumerate(col_titles):
                axes[0, ci].set_title(t, fontsize=11, color=FG, fontweight="bold")

        axes[row_i, 0].set_ylabel(rd["label"], fontsize=10, color=FG, fontweight="bold")
        if rd["mstr"] is not None:
            axes[row_i, 1].set_xlabel(rd["mstr"], fontsize=7.5, color=DIM, ha="center")

    _add_colorbar(fig, axes[:, 1], pet_vmin, pet_vmax, TAU_CMAP, "SUVR")
    _add_colorbar(fig, axes[:, 2], 0, err_vmax, "Oranges", "Abs Error (SUVR)")

    fig.suptitle(f"{mode.upper()}  —  Subject Slices  (ranked by {_RANK_LABEL.get(rank_by, rank_by)})",
                 fontsize=12, color=FG, fontweight="bold")

    out = os.path.join(output_dir, "slice_B.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


def figure_b_scatter(real_vols, gen_vols, pet_vmax, mode, output_dir,
                     all_metrics, ranked, rank_by="ssim"):
    """Voxel density scatter for best/median/worst/pop-mean subjects."""
    real_arr  = np.stack(real_vols)
    gen_arr   = np.stack(gen_vols)
    rows_spec = _build_rows(real_vols, gen_vols, real_arr, gen_arr, all_metrics, ranked)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes_flat = axes.ravel()

    for ax, rd in zip(axes_flat, rows_spec):
        mask  = rd["r"] > 0
        x     = rd["r"][mask].ravel()
        y     = rd["g"][mask].ravel()
        rho, _ = pearsonr(x, y) if x.size > 1 else (float("nan"), None)

        hb = ax.hexbin(x, y, gridsize=80, cmap="hot",
                       extent=[0, pet_vmax, 0, pet_vmax],
                       norm=mcolors.LogNorm(vmin=1))
        ax.plot([0, pet_vmax], [0, pet_vmax], "w--", linewidth=0.9, alpha=0.5)
        ax.set_xlim(0, pet_vmax); ax.set_ylim(0, pet_vmax)
        ax.set_xlabel("Real SUVR", fontsize=9, color=FG)
        ax.set_ylabel("Generated SUVR", fontsize=9, color=FG)
        ax.set_title(rd["label"], fontsize=11, color=FG, fontweight="bold")
        ax.text(0.05, 0.95, f"Pearson r = {rho:.3f}\nn = {x.size:,} voxels",
                transform=ax.transAxes, fontsize=8, color=FG, va="top",
                family="monospace")
        if rd["mstr"] is not None:
            ax.text(0.05, 0.78, rd["mstr"], transform=ax.transAxes,
                    fontsize=7, color=DIM, va="top", family="monospace")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Count (log)", color=DIM, fontsize=7)
        cb.ax.yaxis.set_tick_params(color=DIM, labelcolor=DIM, labelsize=6)
        cb.outline.set_edgecolor("#2a2a2a")

    fig.suptitle(f"{mode.upper()}  —  Subject Voxel Scatter  (ranked by {_RANK_LABEL.get(rank_by, rank_by)})",
                 fontsize=12, color=FG)
    plt.tight_layout()

    out = os.path.join(output_dir, "scatter_B.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_mode(mode, arch, output_dir, use_dk_mask=True, data=None, rank_by="ssim"):
    """Render all glass/slice/scatter figures for one mode.

    data: optional precomputed tuple
        (real_vols, gen_vols, real_norm_list, gen_norm_list, test_ds)
        — same form _load_test_data() returns. When provided, the cache reload is
        skipped (lets evaluate_final.py pass its in-memory SUVR/normalized arrays).
    """
    print(f"\n=== glass_brain_final: mode={mode}  arch={arch}  use_dk_mask={use_dk_mask} ===")
    if data is not None:
        real_vols, gen_vols, real_norm_list, gen_norm_list, test_ds = data
    else:
        real_vols, gen_vols, real_norm_list, gen_norm_list, test_ds = \
            _load_test_data(mode, arch, use_dk_mask)
    if not real_vols:
        print("  No subjects loaded — check generated/ directory.")
        return

    # Save into a per-ranking-metric subdir so different rankings don't overwrite each other.
    output_dir = os.path.join(output_dir, rank_by)
    os.makedirs(output_dir, exist_ok=True)

    affine = _compute_affine(test_ds, real_vols[0].shape)
    pet_vmin, pet_vmax, _, err_vmax = compute_suvr_range(real_vols, gen_vols)
    print(f"  SUVR range: [{pet_vmin:.3f}, {pet_vmax:.3f}]   error vmax: {err_vmax:.3f}")

    all_metrics = [_compute_metrics(r, g) for r, g in zip(real_norm_list, gen_norm_list)]
    ssims       = [m["ssim"] for m in all_metrics]
    ranked      = _rank_subjects(all_metrics, by=rank_by)   # Best/Median/Worst by rank_by

    # ── Print ranking for cross-check with evaluate_final.py ──────────────────
    print(f"\n  {_RANK_LABEL.get(rank_by, rank_by)} ranking (compare with evaluate_final.py to verify subject alignment):")
    for label, idx in ranked.items():
        m = all_metrics[idx]
        print(f"    {label:6s}: subject_{idx:03d}  "
              f"SSIM={m['ssim']:.4f}  Pearson={m['pearson']:.4f}  "
              f"NRMSE={m['nrmse']:.4f}  MSE={m['mse']:.5f}")
    arr = np.array(ssims)
    print(f"  Population:  SSIM={arr.mean():.4f}±{arr.std():.4f}  "
          f"(min={arr.min():.4f}, max={arr.max():.4f})\n")

    print("  Rendering glass brain figures…")
    figure_a_glass(real_vols, gen_vols, real_norm_list, gen_norm_list,
                   mode, output_dir, pet_vmin, pet_vmax, err_vmax, all_metrics, affine)
    figure_b_glass(real_vols, gen_vols, real_norm_list, gen_norm_list,
                   mode, output_dir, pet_vmin, pet_vmax, err_vmax, all_metrics, ranked, affine,
                   rank_by=rank_by)

    print("  Rendering slice figures…")
    figure_a_slice(real_vols, gen_vols, mode, output_dir,
                   pet_vmin, pet_vmax, err_vmax, all_metrics)
    figure_b_slice(real_vols, gen_vols, mode, output_dir,
                   pet_vmin, pet_vmax, err_vmax, all_metrics, ranked, rank_by=rank_by)

    print("  Rendering voxel scatter figures…")
    figure_a_scatter(real_vols, gen_vols, pet_vmax, mode, output_dir)
    figure_b_scatter(real_vols, gen_vols, pet_vmax, mode, output_dir,
                     all_metrics, ranked, rank_by=rank_by)

    print(f"  Done — {mode}.")


def main():
    parser = argparse.ArgumentParser(
        description="TauGenNet glass brain + slice + scatter visualizer."
    )
    parser.add_argument("--mode", choices=["atrophy", "ptau217", "ptau217_mlp", "combined"],
                        default="atrophy")
    parser.add_argument("--arch", default="silu")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Run all three modes sequentially.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-mask", dest="use_dk_mask", action="store_false", default=True,
                        help="Disable DK atlas mask (matches evaluate_final.py --use-mask=False)")
    args = parser.parse_args()

    base_dir = args.output_dir or os.path.join(FIGURES_DIR, args.arch)

    modes = ["atrophy", "ptau217", "combined"] if args.run_all else [args.mode]
    for mode in modes:
        out_dir = os.path.join(base_dir, mode, "glass_brain")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Output → {out_dir}")
        run_mode(mode, args.arch, out_dir, use_dk_mask=args.use_dk_mask)


if __name__ == "__main__":
    main()
