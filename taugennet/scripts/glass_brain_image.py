"""
Combined glass brain + orthogonal slice visualization for TauGenNet.

Produces per mode:
  {mode}_figure_A_glass.png  — population glass brains + per-subject metrics
  {mode}_figure_A_slice.png  — population mean axial slices
  {mode}_figure_B_glass.png  — best / median / worst glass brains + metrics
  {mode}_figure_B_slice.png  — best / median / worst axial slices

Usage:
    python scripts/glass_brain_image.py --mode atrophy
    python scripts/glass_brain_image.py --mode ptau217
    python scripts/glass_brain_image.py --mode combined
    python scripts/glass_brain_image.py --mode atrophy --skip-glass
"""

import argparse
import os
import sys
import tempfile

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from nilearn.plotting import plot_glass_brain
from skimage.metrics import structural_similarity as ssim_fn
from scipy.stats import pearsonr
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import VOL_SHAPE, FIGURES_DIR
from src.dataset_final import unnormalize

# ── design tokens ─────────────────────────────────────────────────────────────
BG      = "#0d0d0d"   # figure background
FG      = "#e8e8e8"   # primary text
DIM     = "#888888"   # secondary / label text
SEP     = "#2a2a2a"   # separator line colour
CBAR_W  = 0.018       # colorbar width fraction
DPI     = 150

# Axial slice z-indices (VOL_SHAPE[2] = 96)
AXIAL_Z = [28, 38, 48, 58, 68]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "text.color":       FG,
    "axes.labelcolor":  FG,
    "axes.edgecolor":   SEP,
    "xtick.color":      FG,
    "ytick.color":      FG,
    "font.family":      "sans-serif",
    "font.size":        10,
    "savefig.facecolor": BG,
    "savefig.edgecolor": "none",
})


# ── metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(real, gen):
    diff = gen - real
    mse  = float(np.mean(diff ** 2))
    mae  = float(np.mean(np.abs(diff)))
    nrmse = float(np.sqrt(mse) / (real.max() - real.min() + 1e-8))
    ssim  = float(ssim_fn(real, gen, data_range=1.0))
    r, _  = pearsonr(real.ravel(), gen.ravel())
    return dict(mse=mse, mae=mae, nrmse=nrmse, ssim=ssim, pearson=float(r))


def _metric_line(m):
    return (f"NRMSE {m['nrmse']:.3f}   MAE {m['mae']:.4f}   "
            f"MSE {m['mse']:.4f}   r {m['pearson']:.3f}   SSIM {m['ssim']:.3f}")


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


def _make_affine(pet_path):
    orig = nib.load(pet_path)
    orig_shape = np.array(orig.shape[:3], dtype=float)
    tgt_shape  = np.array(VOL_SHAPE, dtype=float)
    scale = orig_shape / tgt_shape
    A = orig.affine.copy()
    A[:3, 3] += A[:3, :3] @ ((scale - 1) / 2)
    A[:3, 0] *= scale[0]; A[:3, 1] *= scale[1]; A[:3, 2] *= scale[2]
    return A


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
            print(f"  WARNING: {gen_path} not found, skipping")
            continue
        gen_np = np.load(gen_path) * (real_np > 0)   # mask to brain support

        if hasattr(test_ds, "get_pet_norms"):
            pet_min, pet_max = test_ds.get_pet_norms(i)
        else:
            pet_min, pet_max = _raw_norms(test_ds.pet_paths[i])

        real_norm.append(real_np)
        gen_norm.append(gen_np)
        real_vols.append(unnormalize(real_np, pet_min, pet_max))
        gen_vols.append(unnormalize(gen_np, pet_min, pet_max))

    return real_vols, gen_vols, real_norm, gen_norm, test_ds


# ── glass brain helpers ───────────────────────────────────────────────────────

def _to_nifti(arr, affine):
    return nib.Nifti1Image(arr.astype(np.float32), affine=affine)


def _render_glass(nii_img, cmap, vmin, vmax, threshold=0.0, symmetric_cbar=False):
    """Render a glass brain to RGBA numpy array (on black background)."""
    tmp = None
    fig = plt.figure(figsize=(7, 4), facecolor="black")
    try:
        plot_glass_brain(
            nii_img, figure=fig, display_mode="ortho",
            colorbar=True, cmap=cmap, vmin=vmin, vmax=vmax,
            plot_abs=False, threshold=threshold,
            symmetric_cbar=symmetric_cbar,
            title="", black_bg=True,
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        fig.savefig(tmp, dpi=DPI, bbox_inches="tight", facecolor="black")
        img = plt.imread(tmp)
    finally:
        plt.close(fig)
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    return img


def _safe_vmax(arr, pct=99.5):
    return float(np.percentile(arr, pct))


# ── slice helpers ─────────────────────────────────────────────────────────────

def _axial_slices(vol, z_indices=AXIAL_Z):
    """Return list of 2-D axial slices oriented with anterior at top, left on left."""
    return [np.rot90(vol[:, :, z]) for z in z_indices]


def _add_slice_panel(ax, sl, cmap, vmin, vmax):
    im = ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def _row_colorbar(fig, im, axes_row, label="", orientation="vertical"):
    """Add a single thin colorbar to the right of an axes row."""
    x0 = axes_row[-1].get_position().x1 + 0.005
    y0 = axes_row[-1].get_position().y0
    height = axes_row[0].get_position().y1 - y0
    cax = fig.add_axes([x0, y0, CBAR_W, height])
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    cb.ax.yaxis.set_tick_params(color=DIM, labelcolor=DIM, labelsize=7)
    cb.outline.set_edgecolor(SEP)
    if label:
        cb.set_label(label, color=DIM, fontsize=7)
    return cb


# ── Figure A: glass brain ─────────────────────────────────────────────────────

def figure_a_glass(real_vols, gen_vols, real_norm, gen_norm, affine, mode, output_dir):
    mean_real = np.mean(real_vols, axis=0)
    mean_gen  = np.mean(gen_vols,  axis=0)
    mean_err  = mean_gen - mean_real

    hot_vmax = max(_safe_vmax(mean_real), _safe_vmax(mean_gen))
    err_vmax = float(np.percentile(np.abs(mean_err), 99.5))
    err_thr  = float(np.percentile(np.abs(mean_err), 75))

    specs = [
        (mean_real, "hot",    0,         hot_vmax, 1e-6,    False, "Mean Real PET  (SUVR)"),
        (mean_gen,  "hot",    0,         hot_vmax, 1e-6,    False, "Mean Generated PET  (SUVR)"),
        (mean_err,  "RdBu_r", -err_vmax, err_vmax, err_thr, True,  "Mean Error  Gen − Real"),
    ]
    panels = [_render_glass(_to_nifti(arr, affine), cmap, vmin, vmax, thr, sym)
              for arr, cmap, vmin, vmax, thr, sym, _ in specs]

    all_m = [_compute_metrics(r, g) for r, g in zip(real_norm, gen_norm)]
    keys  = ["nrmse", "mae", "mse", "pearson", "ssim"]
    lbls  = ["NRMSE", "MAE", "MSE", "r", "SSIM"]
    means = {k: np.mean([m[k] for m in all_m]) for k in keys}
    stds  = {k: np.std ([m[k] for m in all_m]) for k in keys}
    summary = "   ".join(f"{l} {means[k]:.3f}±{stds[k]:.3f}" for k, l in zip(keys, lbls))

    fig = plt.figure(figsize=(21, 6.5))
    gs  = mgridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[8, 0.7], hspace=0.06, wspace=0.04,
                             left=0.01, right=0.99, top=0.90, bottom=0.04)

    for col, (panel, (_, _, _, _, _, _, title)) in enumerate(zip(panels, specs)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(panel)
        ax.axis("off")
        ax.set_title(title, fontsize=12, color=FG, pad=8, fontweight="semibold")

    ax_m = fig.add_subplot(gs[1, :])
    ax_m.axis("off")
    ax_m.text(0.5, 0.5, f"Per-subject mean ± std  (n={len(real_norm)}, normalized space)   |   {summary}",
              ha="center", va="center", fontsize=10, color=DIM,
              transform=ax_m.transAxes)

    fig.suptitle(f"{mode.upper()}  —  Population Level", fontsize=14, color=FG, y=0.97)

    out = os.path.join(output_dir, f"{mode}_figure_A_glass.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure A: slices ──────────────────────────────────────────────────────────

def figure_a_slice(real_vols, gen_vols, real_norm, gen_norm, mode, output_dir):
    mean_real = np.mean(real_vols, axis=0)
    mean_gen  = np.mean(gen_vols,  axis=0)
    mean_err  = mean_gen - mean_real

    hot_vmax = max(_safe_vmax(mean_real), _safe_vmax(mean_gen))
    err_vmax = float(np.percentile(np.abs(mean_err), 99.5))

    nz = len(AXIAL_Z)
    fig, axes = plt.subplots(3, nz, figsize=(2.8 * nz, 7.5),
                             gridspec_kw=dict(hspace=0.03, wspace=0.03,
                                              left=0.07, right=0.92,
                                              top=0.91, bottom=0.03))

    row_specs = [
        (mean_real, "hot",    0,         hot_vmax, "Real PET"),
        (mean_gen,  "hot",    0,         hot_vmax, "Generated"),
        (mean_err,  "RdBu_r", -err_vmax, err_vmax, "Error  Gen − Real"),
    ]

    last_im = [None, None, None]
    for row, (vol, cmap, vmin, vmax, row_lbl) in enumerate(row_specs):
        slices = _axial_slices(vol)
        for col, (ax, sl) in enumerate(zip(axes[row], slices)):
            im = _add_slice_panel(ax, sl, cmap, vmin, vmax)
            last_im[row] = im
            if row == 0:
                ax.set_title(f"z = {AXIAL_Z[col]}", fontsize=8, color=DIM, pad=3)
        axes[row, 0].set_ylabel(row_lbl, fontsize=11, color=FG, labelpad=6)

    # One colorbar per row, placed to the right
    fig.canvas.draw()
    for row, im in enumerate(last_im):
        _row_colorbar(fig, im, axes[row], label="SUVR" if row < 2 else "Δ SUVR")

    all_m = [_compute_metrics(r, g) for r, g in zip(real_norm, gen_norm)]
    keys  = ["nrmse", "mae", "mse", "pearson", "ssim"]
    lbls  = ["NRMSE", "MAE", "MSE", "r", "SSIM"]
    means = {k: np.mean([m[k] for m in all_m]) for k in keys}
    stds  = {k: np.std ([m[k] for m in all_m]) for k in keys}
    summary = "   ".join(f"{l} {means[k]:.3f}±{stds[k]:.3f}" for k, l in zip(keys, lbls))

    fig.suptitle(
        f"{mode.upper()}  —  Population Slices  (n={len(real_vols)})\n"
        f"{summary}",
        fontsize=11, color=FG, y=0.99, linespacing=1.6,
    )
    out = os.path.join(output_dir, f"{mode}_figure_A_slice.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure B: glass brain ─────────────────────────────────────────────────────

def _rank_subjects(real_vols, gen_vols):
    mse_vals   = [float(np.mean((r - g) ** 2)) for r, g in zip(real_vols, gen_vols)]
    sorted_idx = np.argsort(mse_vals)
    n = len(sorted_idx)
    return {"Best": int(sorted_idx[0]),
            "Median": int(sorted_idx[n // 2]),
            "Worst":  int(sorted_idx[-1])}


def figure_b_glass(real_vols, gen_vols, real_norm, gen_norm, affine, mode, output_dir):
    subjects = _rank_subjects(real_vols, gen_vols)

    fig = plt.figure(figsize=(22, 18))
    # 3 subject rows × (3 glass brain cols + 1 narrow metrics col)
    outer_gs = mgridspec.GridSpec(3, 2, figure=fig,
                                  width_ratios=[5, 1],
                                  hspace=0.12, wspace=0.04,
                                  left=0.01, right=0.98, top=0.95, bottom=0.02)

    col_titles = ["Real PET  (SUVR)", "Generated PET  (SUVR)", "Error  Gen − Real"]

    for row, (label, idx) in enumerate(subjects.items()):
        r_s, g_s = real_vols[idx], gen_vols[idx]
        err = g_s - r_s
        hot_vmax = max(_safe_vmax(r_s), _safe_vmax(g_s))
        err_vmax = float(np.percentile(np.abs(err), 99.5))
        err_thr  = float(np.percentile(np.abs(err), 75))

        specs = [
            (r_s,  "hot",    0,         hot_vmax, 1e-6,    False),
            (g_s,  "hot",    0,         hot_vmax, 1e-6,    False),
            (err,  "RdBu_r", -err_vmax, err_vmax, err_thr, True),
        ]

        inner_gs = mgridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer_gs[row, 0], wspace=0.03)

        for col, (arr, cmap, vmin, vmax, thr, sym) in enumerate(specs):
            panel = _render_glass(_to_nifti(arr, affine), cmap, vmin, vmax, thr, sym)
            ax = fig.add_subplot(inner_gs[col])
            ax.imshow(panel)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, color=FG,
                             pad=6, fontweight="semibold")

        # Metrics side panel
        m = _compute_metrics(real_norm[idx], gen_norm[idx])
        ax_m = fig.add_subplot(outer_gs[row, 1])
        ax_m.axis("off")
        ax_m.text(0.12, 0.90, label,
                  transform=ax_m.transAxes, fontsize=14,
                  fontweight="bold", color=FG, va="top")
        ax_m.text(0.12, 0.72, _metric_block(m),
                  transform=ax_m.transAxes, fontsize=9.5,
                  color=DIM, va="top", family="monospace",
                  linespacing=1.8)

    fig.suptitle(f"{mode.upper()}  —  Subject Level  (best / median / worst by SUVR MSE)",
                 fontsize=14, color=FG, y=0.98)
    out = os.path.join(output_dir, f"{mode}_figure_B_glass.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure B: slices ──────────────────────────────────────────────────────────

def figure_b_slice(real_vols, gen_vols, real_norm, gen_norm, mode, output_dir):
    subjects = _rank_subjects(real_vols, gen_vols)
    nz = len(AXIAL_Z)
    n_subj = len(subjects)      # 3
    rows_per_subj = 3           # real / gen / error

    row_labels  = ["Real PET", "Generated", "Error  Gen−Real"]
    # Add extra height-0 spacer rows between subject blocks for visual separation
    # Layout: [subj0_row0, subj0_row1, subj0_row2, spacer, subj1_row0, ...]
    spacer_h    = 0.3           # relative height of spacer rows
    row_heights = []
    for s_i in range(n_subj):
        row_heights.extend([1.0] * rows_per_subj)
        if s_i < n_subj - 1:
            row_heights.append(spacer_h)
    n_gs_rows = len(row_heights)

    fig = plt.figure(figsize=(2.8 * nz + 3, 3.0 * n_subj * rows_per_subj))
    gs = mgridspec.GridSpec(n_gs_rows, nz + 1, figure=fig,
                            width_ratios=[*([1] * nz), 0.9],
                            height_ratios=row_heights,
                            hspace=0.04, wspace=0.04,
                            left=0.07, right=0.98, top=0.95, bottom=0.02)

    # gs_row maps (subject_index, local_row) → absolute GridSpec row
    def gs_row(s_i, local_row):
        return s_i * (rows_per_subj + 1) + local_row   # +1 for spacer

    # Global shared scales across all three subjects so rows are directly comparable
    all_idxs = list(subjects.values())
    global_hot_vmax = max(
        max(_safe_vmax(real_vols[i]), _safe_vmax(gen_vols[i])) for i in all_idxs
    )
    global_err_vmax = max(
        float(np.percentile(np.abs(gen_vols[i] - real_vols[i]), 99.5)) for i in all_idxs
    )

    first_subj = True
    for s_i, (label, idx) in enumerate(subjects.items()):
        r_s, g_s = real_vols[idx], gen_vols[idx]
        err = g_s - r_s
        m = _compute_metrics(real_norm[idx], gen_norm[idx])

        vol_specs = [
            (r_s,  "hot",    0,                global_hot_vmax),
            (g_s,  "hot",    0,                global_hot_vmax),
            (err,  "RdBu_r", -global_err_vmax, global_err_vmax),
        ]

        # axes[local_row][col] — track during creation for colorbar use
        axes_grid = [[None] * nz for _ in range(rows_per_subj)]
        last_im   = [None] * rows_per_subj

        for local_row, (vol, cmap, vmin, vmax) in enumerate(vol_specs):
            slices = _axial_slices(vol)
            for col, (sl, z) in enumerate(zip(slices, AXIAL_Z)):
                ax = fig.add_subplot(gs[gs_row(s_i, local_row), col])
                im = _add_slice_panel(ax, sl, cmap, vmin, vmax)
                axes_grid[local_row][col] = ax
                last_im[local_row] = im
                if first_subj and local_row == 0:
                    ax.set_title(f"z={z}", fontsize=8, color=DIM, pad=3)
            axes_grid[local_row][0].set_ylabel(
                row_labels[local_row], fontsize=9, color=FG, labelpad=4)

        first_subj = False

        # Metrics block spans all 3 rows in the rightmost column
        ax_m = fig.add_subplot(
            gs[gs_row(s_i, 0):gs_row(s_i, rows_per_subj - 1) + 1, nz])
        ax_m.axis("off")
        ax_m.text(0.10, 0.95, label,
                  transform=ax_m.transAxes, fontsize=13,
                  fontweight="bold", color=FG, va="top")
        ax_m.text(0.10, 0.78, _metric_block(m),
                  transform=ax_m.transAxes, fontsize=9,
                  color=DIM, va="top", family="monospace", linespacing=1.9)

        # Colorbar: one per row, placed right of the last slice column
        fig.canvas.draw()
        for local_row, im in enumerate(last_im):
            if im is None:
                continue
            _row_colorbar(fig, im, axes_grid[local_row],
                          label="SUVR" if local_row < 2 else "Δ SUVR")

    fig.suptitle(f"{mode.upper()}  —  Subject-Level Slices  (best / median / worst)",
                 fontsize=13, color=FG, y=0.97)
    out = os.path.join(output_dir, f"{mode}_figure_B_slice.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Glass brain + slice visualization for TauGenNet.")
    parser.add_argument("--mode", choices=["atrophy", "ptau217", "combined"],
                        default="atrophy")
    parser.add_argument("--arch", default="silu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-glass", action="store_true",
                        help="Skip glass brain figures (regenerate slices only)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(FIGURES_DIR, args.arch, "glass_brain")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=== {args.mode}  arch={args.arch} ===")
    real_vols, gen_vols, real_norm, gen_norm, test_ds = _load_test_data(args.mode, args.arch)
    if not real_vols:
        print("No subjects loaded — exiting.")
        return

    affine = _make_affine(test_ds.pet_paths[0])

    if not args.skip_glass:
        print("Rendering Figure A glass …")
        figure_a_glass(real_vols, gen_vols, real_norm, gen_norm,
                       affine, args.mode, args.output_dir)
        print("Rendering Figure B glass …")
        figure_b_glass(real_vols, gen_vols, real_norm, gen_norm,
                       affine, args.mode, args.output_dir)

    print("Rendering Figure A slices …")
    figure_a_slice(real_vols, gen_vols, real_norm, gen_norm, args.mode, args.output_dir)
    print("Rendering Figure B slices …")
    figure_b_slice(real_vols, gen_vols, real_norm, gen_norm, args.mode, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
