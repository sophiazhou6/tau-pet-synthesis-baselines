"""
Glass brain visualization for TauGenNet silu runs.

Figure A (population): mean real PET | mean generated PET | mean error
                        + population-level mean±std metrics table
Figure B (subject):    best / median / worst subject × real | generated | error
                        + per-subject metrics in a left text column

Usage:
    python scripts/glass_brain.py --mode atrophy
    python scripts/glass_brain.py --mode ptau217
    python scripts/glass_brain.py --mode combined
    python scripts/glass_brain.py --mode atrophy --arch silu --output-dir results/figures/glass_brain
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
from matplotlib.gridspec import GridSpec
from nilearn.plotting import plot_glass_brain
from skimage.metrics import structural_similarity as ssim_fn
from scipy.stats import pearsonr
from tqdm import tqdm

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import VOL_SHAPE, FIGURES_DIR
from src.dataset_final import unnormalize


# ── metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(real_np, gen_np):
    """All five metrics in normalized [0,1] space (consistent with evaluate.py)."""
    diff     = gen_np - real_np
    mse      = float(np.mean(diff ** 2))
    mae      = float(np.mean(np.abs(diff)))
    nrmse    = float(np.sqrt(mse) / (real_np.max() - real_np.min() + 1e-8))
    ssim_val = float(ssim_fn(real_np, gen_np, data_range=1.0))
    r, _     = pearsonr(real_np.ravel(), gen_np.ravel())
    return {"mse": mse, "mae": mae, "nrmse": nrmse, "ssim": ssim_val, "pearson": float(r)}


def _fmt_metrics(m):
    return (
        f"NRMSE = {m['nrmse']:.3f}\n"
        f"MAE   = {m['mae']:.4f}\n"
        f"MSE   = {m['mse']:.4f}\n"
        f"r     = {m['pearson']:.3f}\n"
        f"SSIM  = {m['ssim']:.3f}"
    )


# ── data loading ─────────────────────────────────────────────────────────────

def _raw_norms(pet_path):
    """Min/max from the original NIfTI for SUVR unnormalization (pre-mask approx)."""
    vol = nib.load(pet_path).get_fdata().astype(np.float32)
    return float(vol.min()), float(vol.max())


def _make_affine(pet_path):
    """
    Compute the MNI affine for the (96,112,96) resampled volume.
    Accounts for the half-voxel origin shift from F.interpolate(align_corners=False).
    """
    orig       = nib.load(pet_path)
    orig_shape = np.array(orig.shape[:3], dtype=float)
    tgt_shape  = np.array(VOL_SHAPE,      dtype=float)
    scale      = orig_shape / tgt_shape
    A          = orig.affine.copy()
    # align_corners=False: output voxel 0 maps to input voxel 0.5*(scale-1)
    A[:3, 3] += A[:3, :3] @ ((scale - 1) / 2)
    A[:3, 0] *= scale[0]
    A[:3, 1] *= scale[1]
    A[:3, 2] *= scale[2]
    return A


def _load_test_data(mode, arch):
    """
    Returns (real_vols, gen_vols, real_norm, gen_norm, test_ds) where:
      real_vols / gen_vols : SUVR float32 arrays, shape (96,112,96)
      real_norm / gen_norm : normalized [0,1] counterparts (for metrics)
    """
    '''
    if mode in ("atrophy", "ptau217"):
        from src.dataset_final import build_dataloaders
        _, _, test_ds, _, _, _ = build_dataloaders(mode=mode, use_dk_mask=True)
    else:
        from src.dataset_combined import build_dataloaders
        # use_mask=True required to match roi_table.py evaluation
        _, _, test_ds, _, _, _ = build_dataloaders(mode="combined", use_dk_mask=True)

    gen_dir = os.path.join(ROOT, "results", "generated", arch, mode)
    n = len(test_ds)

    real_vols, gen_vols = [], []
    real_norm, gen_norm = [], []
    for i in tqdm(range(n), desc=f"Loading {mode}"):
        pet, _, _ = test_ds[i]               # also populates _pet_norms[i] for v2 datasets
        real_np   = pet.squeeze(0).numpy()   # (H,W,D) normalized [0,1]

        gen_path = os.path.join(gen_dir, f"subject_{i:03d}.npy")
        if not os.path.exists(gen_path):
            print(f"  WARNING: {gen_path} not found, skipping subject {i}")
            continue

        gen_np = np.load(gen_path)

        # For atrophy/ptau217 use post-mask norms (populated by __getitem__ above).
        # For combined, dataset has no get_pet_norms(); fall back to raw NIfTI norms.
        if hasattr(test_ds, "get_pet_norms"):
            pet_min, pet_max = test_ds.get_pet_norms(i)
        else:
            pet_min, pet_max = _raw_norms(test_ds.pet_paths[i])

        real_norm.append(real_np)
        gen_norm.append(gen_np)
        real_vols.append(unnormalize(real_np, pet_min, pet_max))
        gen_vols.append(unnormalize(gen_np,   pet_min, pet_max))

    return real_vols, gen_vols, real_norm, gen_norm, test_ds
'''
def _load_test_data(mode, arch):
    """
    Returns (real_vols, gen_vols, real_norm, gen_norm, test_ds) where:
      real_vols / gen_vols : SUVR float32 arrays, shape (96,112,96)
      real_norm / gen_norm : normalized [0,1] counterparts (for metrics)
    """
    if mode in ("atrophy", "ptau217"):
        from src.dataset_final import build_dataloaders
        _, _, test_ds, _, _, _ = build_dataloaders(mode=mode, use_dk_mask=True)
    else:
        from src.dataset_combined import build_dataloaders
        _, _, test_ds, _, _, _ = build_dataloaders(mode="combined", use_dk_mask=True)

    gen_dir = os.path.join(ROOT, "results", "generated", arch, mode)
    n = len(test_ds)

    real_vols, gen_vols = [], []
    real_norm, gen_norm = [], []

    for i in tqdm(range(n), desc=f"Loading {mode}"):
        pet, _, _ = test_ds[i]
        real_np = pet.squeeze(0).numpy()

        gen_path = os.path.join(gen_dir, f"subject_{i:03d}.npy")
        if not os.path.exists(gen_path):
            print(f"  WARNING: {gen_path} not found, skipping subject {i}")
            continue

        gen_np = np.load(gen_path)

        if hasattr(test_ds, "get_pet_norms"):
            pet_min, pet_max = test_ds.get_pet_norms(i)
        else:
            pet_min, pet_max = _raw_norms(test_ds.pet_paths[i])

        # --------------------------------------------------
        # DEBUG FIRST SUBJECT
        # --------------------------------------------------
        if i == 0:
            print("\n========== SUBJECT 0 ==========")

            print("Normalized real")
            print(
                "min =", real_np.min(),
                "max =", real_np.max(),
                "mean =", real_np.mean()
            )

            print("Normalized generated")
            print(
                "min =", gen_np.min(),
                "max =", gen_np.max(),
                "mean =", gen_np.mean()
            )

            real_suvr = unnormalize(real_np, pet_min, pet_max)
            gen_suvr  = unnormalize(gen_np, pet_min, pet_max)

            print("\nSUVR real")
            print(
                "min =", real_suvr.min(),
                "max =", real_suvr.max(),
                "mean =", real_suvr.mean()
            )

            print("SUVR generated")
            print(
                "min =", gen_suvr.min(),
                "max =", gen_suvr.max(),
                "mean =", gen_suvr.mean()
            )

            print("\nPET scaling")
            print("pet_min =", pet_min)
            print("pet_max =", pet_max)

            print("Mean abs error (subject 0) =",
                  np.mean(np.abs(gen_suvr - real_suvr)))

            print("================================\n")

        real_norm.append(real_np)
        gen_norm.append(gen_np)

        real_vols.append(unnormalize(real_np, pet_min, pet_max))
        gen_vols.append(unnormalize(gen_np, pet_min, pet_max))

    # --------------------------------------------------
    # POPULATION DEBUG
    # --------------------------------------------------
    mean_real = np.mean(real_vols, axis=0)
    mean_gen  = np.mean(gen_vols, axis=0)

    err = mean_gen - mean_real

    print("\n========== POPULATION ==========")

    print("Mean real volume")
    print(
        "min =", mean_real.min(),
        "max =", mean_real.max(),
        "mean =", mean_real.mean()
    )

    print("Mean generated volume")
    print(
        "min =", mean_gen.min(),
        "max =", mean_gen.max(),
        "mean =", mean_gen.mean()
    )

    print("Mean error volume")
    print(
        "min =", err.min(),
        "max =", err.max(),
        "mean =", err.mean()
    )

    print("Mean absolute error volume =",
          np.mean(np.abs(err)))

    print("================================\n")

    return real_vols, gen_vols, real_norm, gen_norm, test_ds

# ── glass brain rendering ─────────────────────────────────────────────────────

def _to_nifti(arr, affine):
    return nib.Nifti1Image(arr.astype(np.float32), affine=affine)


def _render(nii_img, cmap, vmin, vmax, title, black_bg=False):
    """Render a glass brain to a numpy image array via a temp file."""
    tmp = None
    fig = plt.figure(figsize=(7, 4))
    try:
        plot_glass_brain(
            nii_img,
            figure=fig,
            display_mode="ortho",
            colorbar=True,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            plot_abs=False,          # signed display — fixes the all-red error panel
            threshold=thr,           # hide near-zero voxels so structure shows through
            symmetric_cbar=True,     # for the error panel
            title=title,
            black_bg=black_bg,
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        fig.savefig(tmp, dpi=150, bbox_inches="tight")
        img = plt.imread(tmp)
    finally:
        plt.close(fig)
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    return img


def _safe_vmax(arr, pct=99.5):
    """Robust colorscale max — clips outliers at the given percentile."""
    return float(np.percentile(arr, pct))


# ── figure A: population level ────────────────────────────────────────────────

def figure_a(real_vols, gen_vols, real_norm, gen_norm, affine, mode, output_dir):
    mean_real = np.mean(real_vols, axis=0)
    mean_gen  = np.mean(gen_vols,  axis=0)
    mean_err  = mean_gen - mean_real

    hot_vmax = max(_safe_vmax(mean_real), _safe_vmax(mean_gen))
    err_vmax = float(np.percentile(np.abs(mean_err), 99.5))

    # Glass brains visualise POOLED volumes (voxelwise mean across all test subjects).
    specs = [
        (mean_real, "hot",    0,         hot_vmax, "Mean Real PET (SUVR)\n[pooled mean volume]"),
        (mean_gen,  "hot",    0,         hot_vmax, "Mean Generated PET (SUVR)\n[pooled mean volume]"),
        (mean_err,  "RdBu_r", -err_vmax, err_vmax, "Mean Error: Generated − Real\n[pooled mean volume]"),
    ]

    panels = [_render(_to_nifti(arr, affine), cmap, vmin, vmax, "")
              for arr, cmap, vmin, vmax, title in specs]

    # Metrics are PER-SUBJECT: computed individually on each subject, then aggregated.
    # These are NOT metrics computed on the pooled mean volumes above.
    all_metrics = [_compute_metrics(r, g) for r, g in zip(real_norm, gen_norm)]
    keys   = ["nrmse", "mae", "mse", "pearson", "ssim"]
    labels = ["NRMSE", "MAE", "MSE", "Pearson r", "SSIM"]
    means  = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    stds   = {k: float(np.std( [m[k] for m in all_metrics])) for k in keys}
    summary = "  ".join(f"{lbl} = {means[k]:.3f} ± {stds[k]:.3f}"
                        for k, lbl in zip(keys, labels))

    fig = plt.figure(figsize=(21, 7))
    gs  = GridSpec(2, 3, figure=fig, height_ratios=[6, 1], hspace=0.05)

    for col, (panel, (_, _, _, _, title)) in enumerate(zip(panels, specs)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(panel)
        ax.set_title(title, fontsize=12, pad=6)
        ax.axis("off")

    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis("off")
    ax_txt.text(
        0.5, 0.5,
        f"Per-subject metrics — mean ± std across n={len(real_norm)} subjects (normalized space)"
        f" — computed per subject, NOT on pooled mean volumes\n{summary}",
        ha="center", va="center", fontsize=11,
        transform=ax_txt.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#aaaaaa"),
    )

    fig.suptitle(
        f"Figure A — Population Level  ({mode}, n={len(real_vols)})"
        f"  |  glass brains: pooled mean volumes  |  metrics: per-subject aggregated",
        fontsize=13, y=1.01,
    )
    out = os.path.join(output_dir, f"{mode}_figure_A.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure B: subject level ───────────────────────────────────────────────────

def figure_b(real_vols, gen_vols, real_norm, gen_norm, affine, mode, output_dir):
    # rank by SUVR-space MSE — visually meaningful and consistent with the error panel
    mse_vals   = [float(np.mean((r - g) ** 2)) for r, g in zip(real_vols, gen_vols)]
    sorted_idx = np.argsort(mse_vals)
    n          = len(sorted_idx)
    subjects   = {
        "Best":   int(sorted_idx[0]),
        "Median": int(sorted_idx[n // 2]),
        "Worst":  int(sorted_idx[-1]),
    }

    # Glass brains and metrics are both PER-SUBJECT throughout Figure B.
    col_titles = ["Real PET (SUVR)\n[per subject]",
                  "Generated PET (SUVR)\n[per subject]",
                  "Error (Gen − Real)\n[per subject]"]

    # 3 rows × 4 cols: col 0 = text label, cols 1-3 = glass brain panels
    fig = plt.figure(figsize=(26, 18))
    gs  = GridSpec(3, 4, figure=fig, width_ratios=[1.1, 3, 3, 3],
                   wspace=0.03, hspace=0.08)

    for row, (label, idx) in enumerate(subjects.items()):
        r_suvr, g_suvr = real_vols[idx], gen_vols[idx]
        err = g_suvr - r_suvr
        hot_vmax = max(_safe_vmax(r_suvr), _safe_vmax(g_suvr))
        err_vmax = float(np.percentile(np.abs(err), 99.5))

        specs = [
            (r_suvr, "hot",    0,         hot_vmax),
            (g_suvr, "hot",    0,         hot_vmax),
            (err,    "RdBu_r", -err_vmax, err_vmax),
        ]

        for col, (arr, cmap, vmin, vmax) in enumerate(specs):
            panel = _render(_to_nifti(arr, affine), cmap, vmin, vmax, "")
            ax    = fig.add_subplot(gs[row, col + 1])
            ax.imshow(panel)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=13, pad=6)

        # text column: subject label + per-subject metrics (normalized space)
        m      = _compute_metrics(real_norm[idx], gen_norm[idx])
        ax_txt = fig.add_subplot(gs[row, 0])
        ax_txt.axis("off")
        ax_txt.text(
            0.5, 0.5,
            f"$\\bf{{{label}}}$\n(per-subject)\n\n{_fmt_metrics(m)}",
            ha="center", va="center", fontsize=11, family="monospace",
            transform=ax_txt.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#bbbbbb"),
        )

    fig.suptitle(
        f"Figure B — Subject Level  ({mode})"
        f"  |  glass brains: per subject  |  metrics: per subject (normalized space)",
        fontsize=13, y=1.005,
    )
    out = os.path.join(output_dir, f"{mode}_figure_B.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Glass brain plots for TauGenNet silu runs.")
    parser.add_argument("--mode",       choices=["atrophy", "ptau217", "combined"],
                        default="atrophy")
    parser.add_argument("--arch",       default="silu",
                        help="Architecture subdirectory under results/generated/")
    parser.add_argument("--output-dir", default=os.path.join(FIGURES_DIR, "glass_brain"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n=== Glass brain: mode={args.mode}  arch={args.arch} ===")

    real_vols, gen_vols, real_norm, gen_norm, test_ds = _load_test_data(args.mode, args.arch)
    if len(real_vols) == 0:
        print("No subjects loaded — exiting.")
        return

    affine = _make_affine(test_ds.pet_paths[0])

    print("Rendering Figure A …")
    figure_a(real_vols, gen_vols, real_norm, gen_norm, affine, args.mode, args.output_dir)

    print("Rendering Figure B …")
    figure_b(real_vols, gen_vols, real_norm, gen_norm, affine, args.mode, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
