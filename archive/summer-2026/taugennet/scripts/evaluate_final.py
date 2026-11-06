#!/usr/bin/env python3
"""
evaluate_final.py — Canonical TauGenNet evaluation script.

Metric methodology (paper-faithful):
  - ALL primary metrics are computed in SUVR (unnormalized) space, DK86-masked.
    Per-subject SSIM uses the per-subject real SUVR range as its data_range.
  - Per-subject ROI metrics (all 5) computed per-subject then averaged — never pooled.
  - Plasma × region NRMSE/MSE/MAE/Pearson in SUVR space (eqs 16-18), for ptau217/combined.
  - The per-subject [0,1] NORMALIZED diagnostics (M1n regional Pearson, M2 voxelwise
    cross-subject r, normalized-vs-SUVR 2×2 summary) are opt-in via
    --normalized-supplementary (default off).

Brain slice plots:
  - All PET panels displayed in SUVR (unnormalized).
  - Real and Generated share a single vmin/vmax derived from the 99th-percentile
    of all real brain voxels in the test set.
  - Error panels share a separate vmax (95th-percentile of all per-voxel errors).
  - Colormap: blue → pink (TAU_CMAP) for PET, Oranges for error.

Usage:
  python scripts/evaluate_final.py --mode atrophy
  python scripts/evaluate_final.py --mode ptau217 --checkpoint-dir results/checkpoints/ptau217
  python scripts/evaluate_final.py --mode combined --arch relu
"""

import os
import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

import torch.nn.functional as F

from src.config import DEVICE, VOL_SHAPE, FIGURES_DIR
from src import dataset_final as _dataset_v2
from src import dataset_combined as _dataset_combined
from src.dataset_final import unnormalize as _unnormalize
from src.dataset_final import REGION_COLS
from src.atlas_labels import load_atlas, get_default_atlas_path
from src.diffusion import DiffusionSchedule
from src.inference import load_models, synthesize_tau_pet, synthesize_no_mri

# 86 DK region names in atlas label order (1-86), stripped of the _ATROPHY_Z suffix.
REGION_NAMES = [c.replace("_ATROPHY_Z", "") for c in REGION_COLS]


# ── colormap ──────────────────────────────────────────────────────────────────
# Blue (low uptake / background) → pink/magenta (high uptake).
TAU_CMAP = LinearSegmentedColormap.from_list(
    "tau_pet",
    ["#08306b", "#2171b5", "#9ecae1", "#fbb4b9", "#f768a1", "#ae017e"],
)

# ── ROI definitions ───────────────────────────────────────────────────────────
ROI_DEFS = {
    "Parahippocampal":   (0.40, 0.55, 0.35, 0.65, 0.30, 0.55),
    "Fusiform":          (0.45, 0.60, 0.30, 0.70, 0.25, 0.50),
    "Inferior Temporal": (0.30, 0.55, 0.25, 0.75, 0.20, 0.55),
    "Hippocampus":       (0.42, 0.52, 0.40, 0.60, 0.35, 0.50),
    "Post. Cingulate":   (0.40, 0.55, 0.38, 0.62, 0.50, 0.70),
    "Entorhinal":        (0.43, 0.55, 0.38, 0.62, 0.28, 0.45),
}
ROI_METRIC_KEYS = ["pearson", "nrmse", "ssim", "mse", "mae"]
# 86-region tables omit SSIM: it is windowed/spatial, so a single region's bounding box
# leaks in neighboring tissue and SSIM is ill-defined for irregular sub-regions. SSIM is
# reported at whole-brain level only (Set 1 + the existing per-subject metrics).
REGION_METRIC_KEYS = ["pearson", "nrmse", "mse", "mae"]

# Plasma p-tau217 intervals (pg/mL).
PLASMA_BINS   = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, float("inf"))]
BIN_LABELS    = ["0-2", "2-4", "4-6", "6-8", "8-10", "10+"]

H, W, D = VOL_SHAPE


# ── helpers ───────────────────────────────────────────────────────────────────

def _plasma_value(cond):
    """Scalar plasma p-tau217 from conditioning tensor.

    ptau217 mode: cond shape (1,) → scalar.
    combined mode: cond shape (87,) = [atrophy z-scores (86) | ptau217 (1)].
    """
    t = cond
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    t = np.asarray(t, dtype=np.float64).ravel()
    return float(t[0]) if t.size == 1 else float(t[-1])


def _safe_ssim(r, g, data_range=1.0):
    win = min(7, min(r.shape))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return float("nan")
    return float(ssim(r, g, data_range=data_range, win_size=win))


def _dk86_mask(test_ds):
    """Shared DK86 region mask (atlas labels 1-86) at VOL_SHAPE as (H,W,D) bool, or None.

    Metrics use this single anatomical mask for every subject instead of per-subject
    real>0, so coverage is identical across subjects and never includes out-of-region
    voxels. None when the dataset was built with --no-use-mask (callers then fall back
    to per-subject real>0).
    """
    if hasattr(test_ds, "_dk_mask"):
        dk = test_ds._dk_mask
        return None if dk is None else (dk.squeeze(0).numpy() > 0)
    # spatial dataset has no shared _dk_mask attr — use the atlas DK86 mask directly,
    # built identically to dataset_final so coverage matches the non-spatial models.
    return _atlas_dk86_binary()


_ATLAS_DK86_BINARY = None


def _atlas_dk86_binary():
    """Global DK86 binary mask (atlas labels 1-86 > 0) at VOL_SHAPE, cached."""
    global _ATLAS_DK86_BINARY
    if _ATLAS_DK86_BINARY is None:
        atlas_data, _, _ = load_atlas(get_default_atlas_path())
        bt = torch.from_numpy((atlas_data > 0).astype(np.float32))[None, None]
        b  = F.interpolate(bt, size=VOL_SHAPE, mode="nearest").squeeze().numpy()
        _ATLAS_DK86_BINARY = b > 0
    return _ATLAS_DK86_BINARY


def _dk86_label_volume(test_ds):
    """Labeled DK86 atlas at VOL_SHAPE as (H,W,D) int array (labels 0-86), or None.

    The dataset only keeps a *binary* DK mask; for per-region (1-86) metrics we need
    the integer labels. Build them the same way the dataset builds its binary mask
    (src/dataset_final.py): nearest-neighbor interpolate the atlas to VOL_SHAPE so the
    labels stay voxel-aligned with the (also nearest/trilinear-resized) data. Returns
    None when the dataset was built with --no-use-mask (no atlas applied).
    """
    if hasattr(test_ds, "_dk_mask") and test_ds._dk_mask is None:
        return None  # non-spatial dataset built with --no-use-mask
    atlas_data, _, _ = load_atlas(get_default_atlas_path())            # (aH,aW,aD) int 0-86
    lab_t = torch.from_numpy(atlas_data.astype(np.float32))[None, None]  # (1,1,aH,aW,aD)
    lab   = F.interpolate(lab_t, size=VOL_SHAPE, mode="nearest").squeeze().numpy()
    return np.rint(lab).astype(np.int32)


def _mask_bbox(mask):
    """Bounding-box slices of a boolean mask (for windowed SSIM)."""
    nz = np.argwhere(mask)
    if nz.size == 0:
        return None
    lo, hi = nz.min(0), nz.max(0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _ssim_brain(real, gen, data_range=None, bbox=None):
    """SSIM on a bounding-box crop, excluding zero padding.

    SSIM is a windowed spatial metric, so it can't be restricted to a 1-D vector of
    in-region voxels like the other metrics. Cropping to the bounding box drops most
    of the zero background — which otherwise inflates SSIM — while preserving the 3-D
    structure the metric needs. Pass ``bbox`` (the DK86 mask box) to score over exactly
    the 86 regions; default falls back to the per-subject real>0 box.

    SSIM is normalization-dependent, so in SUVR space ``data_range`` must reflect the
    real intensity span. When None (the default) it is set per-subject to the crop's
    real max-min — the standard data_range for an unnormalized volume.
    """
    if bbox is None:
        bbox = _mask_bbox(real > 0)
        if bbox is None:
            return float("nan")
    r_crop, g_crop = real[bbox], gen[bbox]
    if data_range is None:
        data_range = float(r_crop.max() - r_crop.min())
        if data_range <= 1e-8:
            return float("nan")
    return _safe_ssim(r_crop, g_crop, data_range=data_range)


def _roi_crop(vol, roi):
    y0, y1, x0, x1, z0, z1 = roi
    return vol[int(y0*H):int(y1*H), int(x0*W):int(x1*W), int(z0*D):int(z1*D)]


def _compute_regional_means(suvr_vols):
    """Return N×K array of per-subject regional mean SUVR values (no masking)."""
    rois = list(ROI_DEFS.keys())
    out  = np.zeros((len(suvr_vols), len(rois)), dtype=np.float64)
    for i, vol in enumerate(suvr_vols):
        for k, roi in enumerate(ROI_DEFS.values()):
            out[i, k] = float(_roi_crop(vol, roi).mean())
    return out


def roi_metrics_per_subject(real_vol, gen_vol, roi):
    """All 5 metrics within an ROI for one subject (voxel-level)."""
    r = _roi_crop(real_vol, roi)
    g = _roi_crop(gen_vol,  roi)
    diff  = g - r
    mse   = float(np.mean(diff ** 2))
    mae   = float(np.mean(np.abs(diff)))
    nrmse = float(np.sqrt(mse) / (r.max() - r.min() + 1e-8))
    r_f, g_f = r.ravel(), g.ravel()
    pearson = float("nan") if r_f.std() < 1e-8 or g_f.std() < 1e-8 else float(pearsonr(r_f, g_f)[0])
    # SUVR-space SSIM: data_range is the per-subject ROI intensity span (not 1.0).
    dr = float(r.max() - r.min())
    ssim_score = _safe_ssim(r, g, data_range=dr) if dr > 1e-8 else float("nan")
    return {"pearson": pearson, "nrmse": nrmse, "ssim": ssim_score, "mse": mse, "mae": mae}


# ── inference ─────────────────────────────────────────────────────────────────

def generate_all(ae, unet, schedule, encode_cond, latent_std, test_ds,
                 cond_mode, n_steps=500, save_dir=None):
    """Run DDPM inference on every test subject.

    Returns
    -------
    real_norm  : list[np.ndarray]  — real PET in [0, 1]  (per-subject min-max)
    gen_masked : list[np.ndarray]  — generated PET in [0, 1], brain-support masked
    gen_raw    : list[np.ndarray]  — generated PET in [0, 1], unmasked
    real_suvr  : list[np.ndarray]  — real PET in SUVR
    gen_suvr   : list[np.ndarray]  — generated PET in SUVR, unmasked (for eq. 18)
    plasma_vals: list[float]|None  — plasma p-tau217 per subject, or None for atrophy mode

    save_dir: if provided, each generated volume is written to
              <save_dir>/subject_{i:03d}.npy for later --use-cached runs.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    real_norm, gen_masked, gen_raw = [], [], []
    real_suvr, gen_suvr = [], []
    plasma_vals = [] if cond_mode in ("ptau217", "combined") else None
    dk_mask = _dk86_mask(test_ds)   # shared DK86 region mask (None → fall back to real>0)

    for i in tqdm(range(len(test_ds)), desc="Generating"):
        pet, mri, cond = test_ds[i]
        real = pet.squeeze().numpy()

        with torch.no_grad():
            gen = synthesize_tau_pet(
                mri.unsqueeze(0).to(DEVICE),
                cond.unsqueeze(0).to(DEVICE),
                ae, unet, schedule, encode_cond, latent_std,
                n_steps=n_steps, sampler="ddpm",
            ).squeeze().cpu().numpy()

        if save_dir:
            out_path = os.path.join(save_dir, f"subject_{i:03d}.npy")
            if os.path.exists(out_path):
                raise FileExistsError(
                    f"Generated file already exists: {out_path}. "
                    "Use --no-save-generated or clear the directory first."
                )
            np.save(out_path, gen)

        # Mask real with the SAME DK86 support as gen. dataset_final pre-masks the real
        # PET with this same shared mask (so this is a no-op there), but dataset_spatial
        # pre-masks with a per-subject valid-voxel mask (broader) — without this, real
        # keeps WM/non-DK signal that gen lacks, which silently craters SSIM (0.88→0.35).
        bm = dk_mask if dk_mask is not None else (real > 0)
        real_norm.append(real * bm)
        gen_raw.append(gen)
        gen_masked.append(gen * bm)

        pmin, pmax = _pet_norms(test_ds, i)
        real_suvr.append(_unnormalize(real, pmin, pmax))
        gen_suvr.append(_unnormalize(gen,  pmin, pmax))

        if plasma_vals is not None:
            plasma_vals.append(_plasma_value(cond))

    if save_dir:
        print(f"Generated volumes cached → {save_dir}")
    return real_norm, gen_masked, gen_raw, real_suvr, gen_suvr, plasma_vals


def _pet_norms(test_ds, i):
    """Per-subject (min, max) used to unnormalize PET back to SUVR.

    dataset_final exposes get_pet_norms(); dataset_combined does not, so fall back to
    the raw NIfTI min/max (matches glass_brain_final.py)."""
    if hasattr(test_ds, "get_pet_norms"):
        return test_ds.get_pet_norms(i)
    vol = nib.load(test_ds.pet_paths[i]).get_fdata().astype(np.float32)
    return float(vol.min()), float(vol.max())


def load_cached_generations(test_ds, cond_mode, gen_dir):
    """Reconstruct generate_all() outputs from cached generations on disk.

    The cached subject_{i:03d}.npy files hold the generated PET in normalized
    [0, 1] space (== gen_raw). Real PET, SUVR, and plasma values are rebuilt from
    test_ds, so the returned tuple is identical in form to generate_all() — letting
    metrics/figures be recomputed without re-running DDPM inference. Subjects with
    no cached file are skipped (consistently across every returned list).
    """
    real_norm, gen_masked, gen_raw = [], [], []
    real_suvr, gen_suvr = [], []
    plasma_vals = [] if cond_mode in ("ptau217", "combined") else None

    n_missing = 0
    dk_mask = _dk86_mask(test_ds)   # shared DK86 region mask (None → fall back to real>0)
    for i in tqdm(range(len(test_ds)), desc="Loading cached"):
        gen_path = os.path.join(gen_dir, f"subject_{i:03d}.npy")
        if not os.path.exists(gen_path):
            n_missing += 1
            continue

        sample = test_ds[i]          # dataset_final: (pet,mri,cond); dataset_spatial: (pet,mri,cond_vec,map,[tissue],diag)
        pet    = sample[0]
        cond   = sample[2] if len(sample) >= 3 else None
        real = pet.squeeze().numpy()
        gen  = np.load(gen_path).astype(np.float32)

        # Mask real with the SAME DK86 support as gen. dataset_final pre-masks the real
        # PET with this same shared mask (so this is a no-op there), but dataset_spatial
        # pre-masks with a per-subject valid-voxel mask (broader) — without this, real
        # keeps WM/non-DK signal that gen lacks, which silently craters SSIM (0.88→0.35).
        bm = dk_mask if dk_mask is not None else (real > 0)
        real_norm.append(real * bm)
        gen_raw.append(gen)
        gen_masked.append(gen * bm)

        pmin, pmax = _pet_norms(test_ds, i)
        real_suvr.append(_unnormalize(real, pmin, pmax))
        gen_suvr.append(_unnormalize(gen,  pmin, pmax))

        if plasma_vals is not None:
            plasma_vals.append(_plasma_value(cond))

    if not gen_raw:
        raise FileNotFoundError(
            f"--use-cached set but no subject_*.npy found in {gen_dir}")
    if n_missing:
        print(f"  WARNING: {n_missing} subject(s) missing from cache, skipped "
              f"(loaded {len(gen_raw)}/{len(test_ds)})")
    return real_norm, gen_masked, gen_raw, real_suvr, gen_suvr, plasma_vals


# ── shared color range ────────────────────────────────────────────────────────

def compute_suvr_range(real_suvr, gen_suvr):
    """Shared vmin/vmax for PET panels and a shared error vmax.

    PET range: 0 to 99th percentile of brain voxels across all real SUVR volumes.
    Error range: 0 to 95th percentile of absolute per-voxel errors across all subjects.
    Both are computed in SUVR space.
    """
    all_brain = np.concatenate([v[v > 0].ravel() for v in real_suvr])
    pet_vmax  = float(np.percentile(all_brain, 99))

    all_err = np.concatenate(
        [np.abs(r - g).ravel() for r, g in zip(real_suvr, gen_suvr)]
    )
    err_vmax = float(np.percentile(all_err, 95))

    return 0.0, pet_vmax, 0.0, err_vmax


# ── whole-brain metrics ────────────────────────────────────────────────────────

def compute_crosssubject_voxelwise_r(real_norm, gen_masked, brain_mask=None, label=""):
    """Voxelwise cross-subject Pearson r map.

    For each voxel (x,y,z), correlates real and generated values across all N
    test subjects. Returns an (H,W,D) map where high values indicate the model
    correctly ranks subjects by tau burden at that location.

    Computed in normalized [0,1] space by default, consistent with wholebrain metrics.
    Brain mask defaults to the union of all subjects' real>0 voxels; pass an explicit
    ``brain_mask`` to reuse identical voxel coverage across calls (e.g. the SUVR variant
    M2s reuses the normalized-space mask so it differs from M2 only in intensity values).
    ``label`` is appended to the printed header to distinguish multiple calls.
    """
    real_stack = np.stack(real_norm,   axis=0).astype(np.float64)  # (N, H, W, D)
    gen_stack  = np.stack(gen_masked,  axis=0).astype(np.float64)

    real_c = real_stack - real_stack.mean(axis=0)
    gen_c  = gen_stack  - gen_stack.mean(axis=0)
    cov    = (real_c * gen_c).mean(axis=0)
    r_map  = (cov / (real_c.std(axis=0) * gen_c.std(axis=0) + 1e-8)).astype(np.float32)

    if brain_mask is None:
        brain_mask = (real_stack > 0).any(axis=0)
    mean_r = float(r_map[brain_mask].mean())
    print(f"Cross-subject voxelwise Pearson r{label}: mean={mean_r:.4f} "
          f"(brain-masked, N={len(real_norm)} subjects)")
    return r_map


def compute_crosssubject_roi_r(r_map):
    """Mean voxelwise cross-subject Pearson r within each bounding-box ROI."""
    roi_r = {}
    for rname, roi in ROI_DEFS.items():
        crop = _roi_crop(r_map, roi)
        roi_r[rname] = float(crop.mean())
        print(f"  {rname:<22s} r = {roi_r[rname]:.4f}")
    return roi_r


def plot_crosssubject_r_map(r_map, real_norm, mode, save_path):
    """Three-panel brain slice figure of the voxelwise cross-subject Pearson r map."""
    brain_mask = np.stack(real_norm, axis=0).any(axis=0)
    display    = np.where(brain_mask, r_map, np.nan)

    H, W, D = display.shape
    slices   = {"Axial": display[H//2, :, :],
                "Sagittal": display[:, W//2, :],
                "Coronal": display[:, :, D//2]}

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
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Cross-subject r map → {save_path}")


def plot_crosssubject_roi_r(roi_r, mode, save_path):
    """Horizontal bar chart of mean cross-subject Pearson r per ROI."""
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
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Cross-subject ROI r → {save_path}")


def compute_wholebrain_metrics(real_norm, gen_masked, brain_mask=None, bbox=None):
    """Per-subject whole-brain metrics in [0, 1] normalized space, in-region only.

    Voxel-wise metrics (Pearson/MSE/MAE/NRMSE) are computed inside ``brain_mask`` —
    the shared DK86 region mask (atlas labels 1-86) when provided, so every subject
    uses identical anatomical coverage and out-of-region/background voxels never
    dilute the score. Falls back to per-subject real>0 when brain_mask is None.
    SSIM uses _ssim_brain() on the region bounding-box crop (``bbox``).
    """
    pearsons, nrmses, ssims, mses, maes = [], [], [], [], []
    for real, gen in zip(real_norm, gen_masked):
        mask  = brain_mask if brain_mask is not None else (real > 0)
        r_in  = real[mask]
        g_in  = gen[mask]
        diff  = g_in - r_in
        mse   = float(np.mean(diff ** 2))
        mae   = float(np.mean(np.abs(diff)))
        nrmse = float(np.sqrt(mse) / (r_in.max() - r_in.min() + 1e-8))
        ssims.append(_ssim_brain(real, gen, bbox=bbox))
        mses.append(mse)
        maes.append(mae)
        nrmses.append(nrmse)
        if r_in.std() < 1e-8 or g_in.std() < 1e-8:
            rho = float("nan")
        else:
            rho, _ = pearsonr(r_in, g_in)
        pearsons.append(float(rho))

    region = "DK86-masked" if brain_mask is not None else "brain-masked"
    print(f"Whole-brain ({region}, SUVR):")
    for name, vals in [("Pearson", pearsons), ("NRMSE", nrmses),
                       ("SSIM", ssims), ("MSE", mses), ("MAE", maes)]:
        a = np.asarray(vals)
        print(f"  {name:<10} {a.mean():.4f} ± {a.std():.4f}")
    return pearsons, nrmses, ssims, mses, maes


# ── canonical metric sets (pooled voxel / per-region / regional cross-subject) ──

def _pooled_voxel_metrics(r, g):
    """{pearson,nrmse,mse,mae} on two already-flattened voxel vectors (pooled).

    NRMSE uses the real range as denominator (consistent with compute_wholebrain_metrics).
    Returns NaNs for degenerate (empty / zero-variance) inputs.
    """
    if r.size == 0:
        return dict(pearson=float("nan"), nrmse=float("nan"), mse=float("nan"), mae=float("nan"))
    diff  = g - r
    mse   = float(np.mean(diff ** 2))
    mae   = float(np.mean(np.abs(diff)))
    rng   = float(r.max() - r.min())
    nrmse = float(np.sqrt(mse) / rng) if rng > 1e-8 else float("nan")
    pearson = float("nan") if (r.std() < 1e-8 or g.std() < 1e-8) else float(pearsonr(r, g)[0])
    return dict(pearson=pearson, nrmse=nrmse, mse=mse, mae=mae)


def compute_pooled_wholebrain_metrics(real_norm, gen_masked, brain_mask):
    """SET 1 — pooled voxel-level whole-brain metrics (normalized [0,1]).

    Pools every in-mask voxel of every subject into one real vector and one generated
    vector, then computes a single Pearson/NRMSE/MSE/MAE over the pool. SSIM cannot be
    pooled into a vector (it is windowed/spatial), so it is the mean of per-subject
    bounding-box SSIM (_ssim_brain), labeled accordingly.
    """
    bbox = _mask_bbox(brain_mask)
    r_all = np.concatenate([real[brain_mask].ravel() for real in real_norm]).astype(np.float64)
    g_all = np.concatenate([gen[brain_mask].ravel()  for gen  in gen_masked]).astype(np.float64)
    m = _pooled_voxel_metrics(r_all, g_all)
    ssims = [_ssim_brain(real, gen, bbox=bbox) for real, gen in zip(real_norm, gen_masked)]
    m["ssim"] = float(np.nanmean(ssims))

    tbl = pd.DataFrame([{k: m[k] for k in ROI_METRIC_KEYS}], index=["Whole brain (pooled)"])
    print(f"\nSET 1 — Pooled voxel-level whole-brain [SUVR, "
          f"N={len(real_norm)} subjects pooled]:")
    for k in ROI_METRIC_KEYS:
        suffix = "  (per-subject avg)" if k == "ssim" else ""
        print(f"  {k.upper():<8} {m[k]:.4f}{suffix}")
    return tbl


def compute_region_pooled_metrics(real_norm, gen_masked, label_vol):
    """SET 2 — pooled voxel-level metrics within each of the 86 DK regions (normalized).

    For each atlas label 1-86, pools that region's voxels across all subjects and computes
    Pearson/NRMSE/MSE/MAE. SSIM is intentionally omitted (whole-brain only — see
    REGION_METRIC_KEYS). Returns an 86-row DataFrame indexed by region name.
    """
    rows = {}
    print("\nSET 2 — Pooled voxel-level per-region (86 DK regions) [SUVR]…")
    for lab, rname in enumerate(REGION_NAMES, start=1):
        region = (label_vol == lab)
        if not region.any():
            rows[rname] = {k: float("nan") for k in REGION_METRIC_KEYS}
            continue
        r_all = np.concatenate([real[region].ravel() for real in real_norm]).astype(np.float64)
        g_all = np.concatenate([gen[region].ravel()  for gen  in gen_masked]).astype(np.float64)
        rows[rname] = _pooled_voxel_metrics(r_all, g_all)
    tbl = pd.DataFrame.from_dict(rows, orient="index")[REGION_METRIC_KEYS]
    print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))
    return tbl


def _regional_means_86(vols, label_vol):
    """N×86 array of per-subject mean intensity within each DK region (label 1-86)."""
    out = np.zeros((len(vols), len(REGION_NAMES)), dtype=np.float64)
    masks = [(label_vol == lab) for lab in range(1, len(REGION_NAMES) + 1)]
    for i, vol in enumerate(vols):
        for k, region in enumerate(masks):
            out[i, k] = float(vol[region].mean()) if region.any() else float("nan")
    return out


def compute_region_crosssubject_metrics(real_norm, gen_masked, label_vol):
    """SET 3 — regional cross-subject metrics over the 86 DK regions (normalized [0,1]).

    Averages each region's voxels to one scalar per subject (N×86 for real and generated),
    then correlates real-vs-generated regional means ACROSS subjects, per region. NRMSE uses
    the mean denominator (consistent with compute_regional_overall). SSIM is omitted (it is
    undefined across N cross-subject scalars; whole-brain only — see REGION_METRIC_KEYS).

    Returns (table_86x4, Rreal, Rgen) where Rreal/Rgen are the N×86 regional-mean matrices.
    """
    Rreal = _regional_means_86(real_norm, label_vol)
    Rgen  = _regional_means_86(gen_masked, label_vol)
    rows = {}
    print("\nSET 3 — Regional cross-subject (86 DK regions) [SUVR, "
          f"N={len(real_norm)} subjects]…")
    for k, rname in enumerate(REGION_NAMES):
        r, g = Rreal[:, k], Rgen[:, k]
        ok = np.isfinite(r) & np.isfinite(g)
        r, g = r[ok], g[ok]
        diff = r - g
        mse  = float(np.mean(diff ** 2)) if r.size else float("nan")
        mae  = float(np.mean(np.abs(diff))) if r.size else float("nan")
        denom = float(r.mean()) if r.size else 0.0
        nrmse = float(np.sqrt(mse) / denom) if abs(denom) > 1e-8 else float("nan")
        pearson = (float(pearsonr(r, g)[0])
                   if r.size >= 2 and r.std() > 1e-8 and g.std() > 1e-8 else float("nan"))
        rows[rname] = dict(pearson=pearson, nrmse=nrmse, mse=mse, mae=mae)
    tbl = pd.DataFrame.from_dict(rows, orient="index")[REGION_METRIC_KEYS]
    print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))
    return tbl, Rreal, Rgen


def save_canonical_metric_sets(set1_tbl, set2_tbl, set3_tbl, Rreal, Rgen, mode, out_dir):
    """Write the three canonical metric sets + the requested N×86 regional-mean CSVs.

    Files (mode-suffixed, in out_dir):
      pooled_wholebrain_{mode}.csv            — SET 1 (1 row)
      region_pooled_{mode}.csv                — SET 2 (86 rows)
      region_crosssubject_metrics_{mode}.csv  — SET 3 (86 rows)
      regional_means_real_{mode}.csv          — N×86, rows=subjects, cols=86 regions
      regional_means_gen_{mode}.csv           — N×86, rows=subjects, cols=86 regions
    """
    os.makedirs(out_dir, exist_ok=True)
    set1_tbl.to_csv(os.path.join(out_dir, f"pooled_wholebrain_{mode}.csv"))
    if set2_tbl is not None:
        set2_tbl.to_csv(os.path.join(out_dir, f"region_pooled_{mode}.csv"))
    if set3_tbl is not None:
        set3_tbl.to_csv(os.path.join(out_dir, f"region_crosssubject_metrics_{mode}.csv"))
    if Rreal is not None:
        idx = [f"subject_{i:03d}" for i in range(Rreal.shape[0])]
        pd.DataFrame(Rreal, index=idx, columns=REGION_NAMES).to_csv(
            os.path.join(out_dir, f"regional_means_real_{mode}.csv"))
        pd.DataFrame(Rgen, index=idx, columns=REGION_NAMES).to_csv(
            os.path.join(out_dir, f"regional_means_gen_{mode}.csv"))
    print(f"Canonical metric sets + regional-mean CSVs → {out_dir}")


# ── ROI per-subject metrics ───────────────────────────────────────────────────

def compute_roi_metrics(real_norm, gen_masked):
    """All 5 metrics × 6 ROIs, per-subject then averaged. Never pooled across subjects."""
    roi_metrics = {r: {k: [] for k in ROI_METRIC_KEYS} for r in ROI_DEFS}

    print("Computing ROI metrics (per-subject → averaged)…")
    for real, gen in tqdm(list(zip(real_norm, gen_masked))):
        for rname, roi in ROI_DEFS.items():
            m = roi_metrics_per_subject(real, gen, roi)
            for k in ROI_METRIC_KEYS:
                roi_metrics[rname][k].append(m[k])

    tables = {k: pd.DataFrame(index=["All subjects"], columns=list(ROI_DEFS.keys()), dtype=float)
              for k in ROI_METRIC_KEYS}
    for rname in ROI_DEFS:
        for k in ROI_METRIC_KEYS:
            vals = [v for v in roi_metrics[rname][k] if not np.isnan(v)]
            tables[k].loc["All subjects", rname] = round(float(np.mean(vals)), 6) if vals else float("nan")

    for k in ROI_METRIC_KEYS:
        print(f"\nROI {k.upper()} [per-subject avg, voxel-level]:")
        print(tables[k].to_string())
    return tables, roi_metrics


# ── ablation (MRI+cond vs cond-only) ─────────────────────────────────────────

def compute_ablation_metrics(ae, unet, schedule, encode_cond, latent_std,
                              test_ds, n_steps=500):
    """ROI metrics for MRI+cond vs cond-only conditions."""
    conds = ["MRI+cond", "cond-only"]
    roi_m = {c: {r: {k: [] for k in ROI_METRIC_KEYS} for r in ROI_DEFS} for c in conds}

    print("Running ablation study…")
    for i in tqdm(range(len(test_ds))):
        pet, mri, cond = test_ds[i]
        pet_np = pet.squeeze().numpy()
        brain  = pet_np > 0

        with torch.no_grad():
            gen_full = synthesize_tau_pet(
                mri.unsqueeze(0).to(DEVICE), cond.unsqueeze(0).to(DEVICE),
                ae, unet, schedule, encode_cond, latent_std,
                n_steps=n_steps, sampler="ddpm",
            ).squeeze().cpu().numpy() * brain

            gen_ablat = synthesize_no_mri(
                mri.unsqueeze(0).to(DEVICE), cond.unsqueeze(0).to(DEVICE),
                ae, unet, schedule, encode_cond, latent_std,
                n_steps=n_steps,
            ).squeeze().cpu().numpy() * brain

        for rname, roi in ROI_DEFS.items():
            m_full  = roi_metrics_per_subject(pet_np, gen_full,  roi)
            m_ablat = roi_metrics_per_subject(pet_np, gen_ablat, roi)
            for k in ROI_METRIC_KEYS:
                roi_m["MRI+cond"][rname][k].append(m_full[k])
                roi_m["cond-only"][rname][k].append(m_ablat[k])

    tables = {k: pd.DataFrame(index=conds, columns=list(ROI_DEFS.keys()), dtype=float)
              for k in ROI_METRIC_KEYS}
    for cond_name in conds:
        for rname in ROI_DEFS:
            for k in ROI_METRIC_KEYS:
                vals = [v for v in roi_m[cond_name][rname][k] if not np.isnan(v)]
                tables[k].loc[cond_name, rname] = round(float(np.mean(vals)), 6) if vals else float("nan")

    for k in ROI_METRIC_KEYS:
        print(f"\nAblation {k.upper()} [per-subject avg, voxel-level]:")
        print(tables[k].to_string())
    return tables


# ── regional-mean SUVR metrics (eq. 16-18 framework) ─────────────────────────

def compute_regional_overall(real_suvr, gen_suvr):
    """Regional-mean SUVR metrics across ALL test subjects — no plasma grouping.

    Computes R_{i,k} per subject (mean SUVR in SUVR space, no masking), then
    NRMSE/MSE/MAE/Pearson across all N subjects treated as one group.

    NRMSE denominator = r.mean() = R̄_real_k (eq. 18 formula, single-bin case).

    Returns (tables, Rreal, Rgen) where Rreal/Rgen are N×K arrays that can be
    passed directly to compute_plasma_tables to avoid recomputation.
    """
    rois  = list(ROI_DEFS.keys())
    Rreal = _compute_regional_means(real_suvr)
    Rgen  = _compute_regional_means(gen_suvr)
    tables = {}

    for metric in ("nrmse", "mse", "mae", "pearson"):
        tbl = pd.DataFrame(index=["All subjects"], columns=rois, dtype=float)
        for k, rname in enumerate(rois):
            r, g = Rreal[:, k], Rgen[:, k]
            diff = r - g
            if metric == "nrmse":
                denom = float(r.mean())
                val = float(np.sqrt(np.mean(diff**2))) / denom if abs(denom) > 1e-8 else float("nan")
            elif metric == "mse":
                val = float(np.mean(diff**2))
            elif metric == "mae":
                val = float(np.mean(np.abs(diff)))
            elif metric == "pearson":
                val = float(pearsonr(r, g)[0]) if r.size >= 2 else float("nan")
            tbl.loc["All subjects", rname] = round(val, 6) if not np.isnan(val) else float("nan")
        tables[metric] = tbl
        print(f"\nRegional-mean SUVR {metric.upper()} [all subjects, eq. 18 framework]:")
        print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))

    return tables, Rreal, Rgen


def compute_regional_overall_normalized(real_norm, gen_raw):
    """SUPPLEMENTARY diagnostic — regional-mean cross-subject Pearson in per-subject
    [0, 1] NORMALIZED space (M1n).

    Identical computation to compute_regional_overall (per-ROI Pearson of per-subject
    regional means across all test subjects), but on the normalized volumes instead of
    SUVR. Per-subject min-max normalization divides out the absolute tau *burden*, so
    this isolates the regional *spatial pattern* — directly comparable to the normalized
    voxelwise cross-subject r (M2).

    NOTE: This is NOT the paper's regional metric. The paper (eqs 16-18) is defined on
    mean tau PET (SUVR) values grouped by plasma interval; that lives in
    compute_regional_overall / compute_plasma_tables and is left untouched.

    Pearson only — NRMSE/MSE/MAE in [0, 1] space are not meaningful here.
    """
    rois  = list(ROI_DEFS.keys())
    Rreal = _compute_regional_means(real_norm)   # N×K, normalized space
    Rgen  = _compute_regional_means(gen_raw)
    tbl = pd.DataFrame(index=["All subjects"], columns=rois, dtype=float)
    for k, rname in enumerate(rois):
        r, g = Rreal[:, k], Rgen[:, k]
        if r.size >= 2 and r.std() > 1e-8 and g.std() > 1e-8:
            val = float(pearsonr(r, g)[0])
        else:
            val = float("nan")
        tbl.loc["All subjects", rname] = round(val, 6) if not np.isnan(val) else float("nan")
    print("\nRegional-mean NORMALIZED [0,1] cross-subject PEARSON "
          "[SUPPLEMENTARY — burden removed, NOT a paper metric]:")
    print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"  mean over ROIs = {np.nanmean(tbl.to_numpy(dtype=float)):.4f}")
    return tbl


def compute_plasma_tables(real_suvr, gen_suvr, plasma_vals, Rreal=None, Rgen=None):
    """G×K tables for NRMSE/MSE/MAE/Pearson on regional-mean SUVR (eq. 18).

    Rreal/Rgen: optional pre-computed N×K regional mean arrays (from
    compute_regional_overall). If None, they are computed here.
    """
    rois = list(ROI_DEFS.keys())
    if Rreal is None:
        Rreal = _compute_regional_means(real_suvr)
    if Rgen is None:
        Rgen  = _compute_regional_means(gen_suvr)

    plasma = np.asarray(plasma_vals, dtype=np.float64)
    tables = {}
    counts = {}

    for metric in ("nrmse", "mse", "mae", "pearson"):
        tbl = pd.DataFrame(index=BIN_LABELS, columns=rois, dtype=float)
        bin_counts = {}
        for (lo, hi), label in zip(PLASMA_BINS, BIN_LABELS):
            mask = (plasma >= lo) & (plasma < hi)
            bin_counts[label] = int(mask.sum())
            for k, rname in enumerate(rois):
                r, g = Rreal[mask, k], Rgen[mask, k]
                if r.size == 0:
                    tbl.loc[label, rname] = float("nan")
                    continue
                diff = r - g
                if metric == "nrmse":
                    denom = float(r.mean())
                    val = float(np.sqrt(np.mean(diff ** 2))) / denom if abs(denom) > 1e-8 else float("nan")
                elif metric == "mse":
                    val = float(np.mean(diff ** 2))
                elif metric == "mae":
                    val = float(np.mean(np.abs(diff)))
                elif metric == "pearson":
                    val = float(pearsonr(r, g)[0]) if r.size >= 2 else float("nan")
                tbl.loc[label, rname] = round(val, 6) if not np.isnan(val) else float("nan")

        tables[metric] = tbl
        counts[metric] = bin_counts
        print(f"\nPlasma × Region {metric.upper()} [eq. 18, SUVR, regional-mean]:")
        print(f"  Subjects per bin: {bin_counts}")
        print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))

    return tables, counts


# ── figures ───────────────────────────────────────────────────────────────────

def _imshow_pet(ax, slc, vmin, vmax):
    """Display a PET slice with the canonical blue→pink colormap."""
    ax.imshow(slc, cmap=TAU_CMAP, vmin=vmin, vmax=vmax, origin="lower")
    ax.axis("off")


def _imshow_err(ax, slc, vmax):
    """Display an absolute-error slice."""
    ax.imshow(slc, cmap="Oranges", vmin=0, vmax=vmax, origin="lower")
    ax.axis("off")


def _add_colorbar(fig, ax, vmin, vmax, cmap, label, shrink=0.8):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=shrink, label=label)


def plot_subject_comparison(real_suvr, gen_suvr, pearsons, nrmses, ssims, mses, maes,
                             mode, pet_vmin, pet_vmax, err_vmax, save_path):
    """Best / median / worst subject (by SSIM) + population mean, in SUVR space."""
    ssim_arr   = np.array(ssims)
    best_idx   = int(np.argmax(ssim_arr))
    worst_idx  = int(np.argmin(ssim_arr))
    median_idx = int(np.argsort(ssim_arr)[len(ssim_arr) // 2])

    real_arr = np.stack(real_suvr)
    gen_arr  = np.stack(gen_suvr)
    mean_r   = real_arr.mean(axis=0)
    mean_g   = gen_arr.mean(axis=0)
    mean_err = np.abs(real_arr - gen_arr).mean(axis=0)

    def _mstr(i):
        return (f"Pearson={pearsons[i]:.3f}  NRMSE={nrmses[i]:.3f}  SSIM={ssims[i]:.3f}"
                f"\nMSE={mses[i]:.4f}  MAE={maes[i]:.4f}")

    rows = [
        (real_suvr[best_idx],   gen_suvr[best_idx],   "Best",   _mstr(best_idx)),
        (real_suvr[median_idx], gen_suvr[median_idx], "Median", _mstr(median_idx)),
        (real_suvr[worst_idx],  gen_suvr[worst_idx],  "Worst",  _mstr(worst_idx)),
        (mean_r,                mean_g,               "Population mean", None),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    col_titles = ["Real PET (SUVR)", "Generated PET (SUVR)", "Abs Error (SUVR)"]

    for row_idx, (real, gen, row_label, mstr) in enumerate(rows):
        diff = mean_err if mstr is None else np.abs(real - gen)
        mid  = real.shape[2] // 2

        _imshow_pet(axes[row_idx, 0], real[:, :, mid], pet_vmin, pet_vmax)
        _imshow_pet(axes[row_idx, 1], gen[:, :, mid],  pet_vmin, pet_vmax)
        _imshow_err(axes[row_idx, 2], diff[:, :, mid], err_vmax)

        if row_idx == 0:
            for ci, t in enumerate(col_titles):
                axes[0, ci].set_title(t, fontsize=11, fontweight="bold")
        # axis("off") in the imshow helpers hides set_ylabel/set_xlabel, so use
        # text artists (unaffected by axis off) for the row label and metrics.
        axes[row_idx, 0].text(-0.06, 0.5, row_label, transform=axes[row_idx, 0].transAxes,
                              rotation=90, va="center", ha="center",
                              fontsize=11, fontweight="bold")
        if mstr is not None:
            axes[row_idx, 1].text(0.5, -0.04, mstr, transform=axes[row_idx, 1].transAxes,
                                  va="top", ha="center", fontsize=8)

    plt.suptitle(
        f"Real vs Generated Tau PET — {mode}  |  "
        f"SSIM={np.mean(ssims):.3f}±{np.std(ssims):.3f}  "
        f"Pearson={np.mean(pearsons):.3f}  MSE={np.mean(mses):.4f}",
        fontsize=11, fontweight="bold")

    # Reserve a strip on the right for two non-overlapping colorbars (dedicated
    # axes — attaching bars to the panel columns + tight_layout makes them overlap).
    fig.subplots_adjust(left=0.07, right=0.85, top=0.95, bottom=0.04,
                        wspace=0.05, hspace=0.18)
    sm_pet = plt.cm.ScalarMappable(cmap=TAU_CMAP, norm=plt.Normalize(pet_vmin, pet_vmax))
    sm_err = plt.cm.ScalarMappable(cmap=plt.cm.Oranges, norm=plt.Normalize(0, err_vmax))
    cax_pet = fig.add_axes([0.87, 0.20, 0.015, 0.60])
    cax_err = fig.add_axes([0.94, 0.20, 0.015, 0.60])
    fig.colorbar(sm_pet, cax=cax_pet, label="SUVR")
    fig.colorbar(sm_err, cax=cax_err, label="Abs Error (SUVR)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Subject comparison → {save_path}")


def plot_population_comparison(real_suvr, gen_suvr, mode,
                                pet_vmin, pet_vmax, err_vmax, save_path):
    """Population mean real / generated / abs error, 3 orthogonal views, SUVR."""
    real_arr = np.stack(real_suvr)
    gen_arr  = np.stack(gen_suvr)
    mean_r   = real_arr.mean(axis=0)
    mean_g   = gen_arr.mean(axis=0)
    mean_err = np.abs(real_arr - gen_arr).mean(axis=0)

    views = {"Axial": 2, "Sagittal": 0, "Coronal": 1}
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))

    for row_idx, (view_name, axis_dim) in enumerate(views.items()):
        mid = mean_r.shape[axis_dim] // 2
        r_slc = np.take(mean_r,   mid, axis=axis_dim)
        g_slc = np.take(mean_g,   mid, axis=axis_dim)
        e_slc = np.take(mean_err, mid, axis=axis_dim)

        _imshow_pet(axes[row_idx, 0], r_slc, pet_vmin, pet_vmax)
        _imshow_pet(axes[row_idx, 1], g_slc, pet_vmin, pet_vmax)
        _imshow_err(axes[row_idx, 2], e_slc, err_vmax)

        if row_idx == 0:
            axes[0, 0].set_title("Mean Real PET (SUVR)",      fontsize=11, fontweight="bold")
            axes[0, 1].set_title("Mean Generated PET (SUVR)", fontsize=11, fontweight="bold")
            axes[0, 2].set_title("Mean Abs Error (SUVR)",     fontsize=11, fontweight="bold")
        # axis("off") hides set_ylabel, so use a text artist for the view label.
        axes[row_idx, 0].text(-0.06, 0.5, view_name, transform=axes[row_idx, 0].transAxes,
                              rotation=90, va="center", ha="center", fontsize=11)

    plt.suptitle(f"Population Mean — {mode}  (n={len(real_suvr)} test subjects)",
                 fontsize=12, fontweight="bold")

    # Reserve a strip on the right for two non-overlapping colorbars (dedicated axes).
    fig.subplots_adjust(left=0.06, right=0.85, top=0.92, bottom=0.03,
                        wspace=0.05, hspace=0.08)
    sm_pet = plt.cm.ScalarMappable(cmap=TAU_CMAP, norm=plt.Normalize(pet_vmin, pet_vmax))
    sm_err = plt.cm.ScalarMappable(cmap=plt.cm.Oranges, norm=plt.Normalize(0, err_vmax))
    cax_pet = fig.add_axes([0.87, 0.20, 0.015, 0.55])
    cax_err = fig.add_axes([0.94, 0.20, 0.015, 0.55])
    fig.colorbar(sm_pet, cax=cax_pet, label="SUVR")
    fig.colorbar(sm_err, cax=cax_err, label="Abs Error (SUVR)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Population comparison → {save_path}")


def plot_wholebrain_boxplot(pearsons, nrmses, ssims, mses, maes, mode, save_path):
    """Boxplot of 5 whole-brain per-subject metrics."""
    metrics = {"Pearson": pearsons, "NRMSE": nrmses, "SSIM": ssims, "MSE": mses, "MAE": maes}
    fig, axes = plt.subplots(1, 5, figsize=(16, 5))
    fig.suptitle(f"Per-subject whole-brain metrics — {mode}  (n={len(pearsons)})",
                 fontsize=13, fontweight="bold")

    rng = np.random.default_rng(0)
    for ax, (name, vals) in zip(axes, metrics.items()):
        v = np.asarray(vals)
        ax.boxplot(v, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#aec6e8", color="#2c5f8a"),
                   medianprops=dict(color="#d62728", linewidth=2),
                   whiskerprops=dict(color="#2c5f8a"), capprops=dict(color="#2c5f8a"),
                   flierprops=dict(marker="o", markersize=4, color="#999999"))
        jitter = rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(1 + jitter, v, alpha=0.55, s=18, color="#1f77b4", zorder=3)
        ax.axhline(v.mean(), color="#ff7f0e", linestyle="--", linewidth=1.2)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_xlabel(f"median {float(np.median(v)):.3f}\nmean {v.mean():.3f} ± {v.std():.3f}",
                      fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Whole-brain boxplot → {save_path}")


def plot_roi_metrics_boxplot(roi_metrics, mode, save_path):
    """One panel per metric, one box per ROI, per-subject distribution."""
    rois = list(ROI_DEFS.keys())
    fig, axes = plt.subplots(len(ROI_METRIC_KEYS), 1,
                             figsize=(12, 3.2 * len(ROI_METRIC_KEYS)))
    fig.suptitle(f"Per-subject ROI metrics — {mode}  [per-subject avg, voxel-level]",
                 fontsize=13, fontweight="bold")

    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, ROI_METRIC_KEYS):
        data = [np.asarray([v for v in roi_metrics[r][metric] if not np.isnan(v)])
                for r in rois]
        ax.boxplot(data, patch_artist=True, widths=0.5,
                   boxprops=dict(facecolor="#aec6e8", color="#2c5f8a"),
                   medianprops=dict(color="#d62728", linewidth=2),
                   whiskerprops=dict(color="#2c5f8a"), capprops=dict(color="#2c5f8a"),
                   flierprops=dict(marker="o", markersize=4, color="#999999"))
        labels = []
        for i, (rname, vals) in enumerate(zip(rois, data)):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(i + 1 + jitter, vals, alpha=0.55, s=14, color="#1f77b4", zorder=3)
            labels.append(f"{rname}\n{vals.mean():.3f}±{vals.std():.3f}" if len(vals) else f"{rname}\nNaN")
        ax.set_xticks(range(1, len(rois) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(metric.upper(), fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROI metrics boxplot → {save_path}")


def plot_pooled_wholebrain_scatter(real_suvr, gen_masked, brain_mask, set1_tbl, mode, save_path):
    """SET 1 figure — pooled in-brain voxel density (real vs generated), R annotated.

    Directly visualizes what pooled_wholebrain_{mode}.csv summarizes: every in-mask voxel
    of every subject pooled into one real/generated pair, in SUVR space. Hexbin (log
    density) because the pool is millions of voxels. ALL pooled voxels are plotted — no
    subsampling — so the figure shows every voxel that enters the statistic; axes span the
    data's SUVR range.
    """
    r_all = np.concatenate([real[brain_mask].ravel() for real in real_suvr]).astype(np.float64)
    g_all = np.concatenate([gen[brain_mask].ravel()  for gen  in gen_masked]).astype(np.float64)
    n = r_all.size
    rp, gp = r_all, g_all

    # Data-driven SUVR axis range (shared, so the y = x identity line is meaningful).
    lo = float(min(rp.min(), gp.min()))
    hi = float(max(rp.max(), gp.max()))

    s = set1_tbl.iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 6))
    hb = ax.hexbin(rp, gp, gridsize=80, bins="log", cmap="viridis",
                   extent=(lo, hi, lo, hi), mincnt=1)
    ax.plot([lo, hi], [lo, hi], color="#d62728", linestyle="--", linewidth=1.2, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Real (SUVR)", fontsize=10)
    ax.set_ylabel("Generated (SUVR)", fontsize=10)
    ax.set_title(f"SET 1 — Pooled whole-brain voxels — {mode}\n"
                 f"R={s['pearson']:.3f}  NRMSE={s['nrmse']:.3f}  "
                 f"MSE={s['mse']:.4f}  MAE={s['mae']:.3f}\n"
                 f"(n={len(real_suvr)} subjects, {n:,} in-brain voxels pooled)",
                 fontsize=9.5, fontweight="bold")
    fig.colorbar(hb, ax=ax, label="log10(voxel count)")
    ax.legend(loc="upper left", fontsize=8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SET 1 pooled-voxel scatter → {save_path}")


def _plot_region_r_bar(pearson_series, title, save_path):
    """Horizontal bar of per-region Pearson R (86 DK regions), sorted ascending so the
    strongest regions sit at the top; negative R bars are colored red."""
    s = pearson_series.dropna().sort_values(ascending=True)
    vals = s.values
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(8, max(12, 0.22 * len(s))))
    ax.barh(range(len(s)), vals, color=colors, height=0.8)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=6)
    ax.set_ylim(-0.5, len(s) - 0.5)
    mean_r = float(np.nanmean(vals))
    ax.axvline(mean_r, color="#ff7f0e", linestyle="--", linewidth=1.3,
               label=f"mean R = {mean_r:.3f}")
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Pearson R", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Region R bar → {save_path}")


def plot_region_pooled_r_bar(set2_tbl, mode, save_path):
    """SET 2 figure — 86-region pooled-voxel Pearson R (matches region_pooled_{mode}.csv)."""
    _plot_region_r_bar(set2_tbl["pearson"],
                       f"SET 2 — Per-region pooled-voxel R (86 DK regions) — {mode}", save_path)


def plot_region_crosssubject_r_bar(set3_tbl, mode, save_path):
    """SET 3 figure — 86-region cross-subject Pearson R (matches
    region_crosssubject_metrics_{mode}.csv)."""
    _plot_region_r_bar(set3_tbl["pearson"],
                       f"SET 3 — Regional cross-subject R (86 DK regions) — {mode}", save_path)


def plot_ablation_comparison(ablation_tables, mode, save_path):
    """Grouped bar chart: MRI+cond vs cond-only per ROI for MSE and Pearson."""
    rois = list(ROI_DEFS.keys())
    x, w = np.arange(len(rois)), 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in [
        (axes[0], "mse",     "MSE",     "Region-wise MSE"),
        (axes[1], "pearson", "Pearson", "Region-wise Pearson"),
    ]:
        tbl = ablation_tables[metric]
        ax.bar(x - w/2, [float(tbl.loc["MRI+cond",  r]) for r in rois], w,
               label="MRI + cond",  color="#4c72b0")
        ax.bar(x + w/2, [float(tbl.loc["cond-only", r]) for r in rois], w,
               label="cond-only",   color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(rois, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{title}\n[per-subject avg, voxel-level]", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle(f"MRI Ablation — Per-ROI Metrics ({mode})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation comparison → {save_path}")


def plot_regional_overall_heatmap(tables, mode, fig_dir):
    """Single combined figure: one row per metric, columns = ROIs.

    Metric order: NRMSE, MSE, MAE, Pearson, SSIM.
    NRMSE/MSE/MAE/Pearson are computed on regional-mean SUVR scalars (eq. 18).
    SSIM is per-ROI crop voxel-level, per-subject averaged, [0,1] space.
    Each row has its own colormap and colorbar so different scales don't clash.
    """
    metric_order = [m for m in ("nrmse", "mse", "mae", "pearson", "ssim") if m in tables]
    cmaps  = {"nrmse": "YlOrRd", "mse": "YlOrRd", "mae": "YlOrRd",
              "pearson": "RdYlGn", "ssim": "RdYlGn"}
    labels = {"nrmse": "NRMSE\n[reg. mean SUVR]", "mse":  "MSE\n[reg. mean SUVR]",
              "mae":   "MAE\n[reg. mean SUVR]",  "pearson": "Pearson\n[reg. mean SUVR]",
              "ssim":  "SSIM\n[voxel crop, [0,1]]"}

    rois = list(tables[metric_order[0]].columns)
    n    = len(metric_order)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, len(rois) * 1.6), 1.9 * n + 1.0))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_order):
        data = tables[metric].values.astype(float).reshape(1, -1)
        im   = ax.imshow(data, cmap=cmaps[metric], aspect="auto",
                         vmin=np.nanmin(data), vmax=np.nanmax(data))
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels([] if metric != metric_order[-1] else rois,
                           rotation=30, ha="right", fontsize=9)
        ax.set_yticks([0])
        ax.set_yticklabels([labels[metric]], fontsize=8)
        for j, v in enumerate(data[0]):
            ax.text(j, 0, "—" if np.isnan(v) else f"{v:.3f}",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="black" if 0.2 < (v - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-9) < 0.8 else "white")
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    fig.suptitle(f"Regional metrics — all test subjects — {mode} mode", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(fig_dir, f"regional_overall_{mode}.png")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Regional overall heatmap → {save_path}")


def plot_plasma_heatmap(table, metric, mode, save_path):
    """Annotated heatmap of a G×K plasma × region metric table."""
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
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label=metric.upper())
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Plasma p-tau217 (pg/mL)")
    ax.set_title(f"Plasma × Region {metric.upper()} [SUVR, eq. 18] — {mode}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plasma heatmap ({metric}) → {save_path}")


def save_tables_to_disk(roi_tables, ablation_tables, plasma_tables, plasma_counts,
                        regional_tables, mode, out_dir,
                        regional_norm_table=None, crosssubject_summary=None):
    """Write all metric tables to CSV and a combined markdown file.

    regional_norm_table   : supplementary M1n table (normalized-space regional Pearson).
    crosssubject_summary  : dict with the M2/M2s scalar means + per-ROI r (supplementary).
    Both are clearly labeled as supplementary (NOT paper metrics).
    """
    os.makedirs(out_dir, exist_ok=True)
    md_lines = [f"# TauGenNet Evaluation — {mode} mode\n"]

    if regional_tables:
        md_lines.append("\n## Regional-mean SUVR [all subjects, eq. 18 framework]\n")
        for k, tbl in regional_tables.items():
            csv_path = os.path.join(out_dir, f"regional_overall_{k}_{mode}.csv")
            tbl.to_csv(csv_path)
            md_lines.append(f"\n### {k.upper()}\n\n")
            md_lines.append(tbl.to_markdown(floatfmt=".4f") + "\n")

    if regional_norm_table is not None:
        csv_path = os.path.join(out_dir, f"regional_overall_pearson_normalized_{mode}.csv")
        regional_norm_table.to_csv(csv_path)
        md_lines.append("\n## [SUPPLEMENTARY] Regional-mean cross-subject PEARSON — "
                        "NORMALIZED [0,1] space (burden removed, NOT a paper metric)\n\n")
        md_lines.append(regional_norm_table.to_markdown(floatfmt=".4f") + "\n")

    if crosssubject_summary is not None:
        md_lines.append("\n## [SUPPLEMENTARY] Cross-subject voxelwise Pearson r — "
                        "2×2 intensity×scale diagnostic\n\n")
        cs_df = pd.DataFrame(crosssubject_summary)
        cs_df.to_csv(os.path.join(out_dir, f"crosssubject_2x2_{mode}.csv"))
        md_lines.append(cs_df.to_markdown(floatfmt=".4f") + "\n")

    for k in ROI_METRIC_KEYS:
        csv_path = os.path.join(out_dir, f"roi_{k}_{mode}.csv")
        roi_tables[k].to_csv(csv_path)
        md_lines.append(f"\n## ROI {k.upper()} [per-subject avg, voxel-level]\n\n")
        md_lines.append(roi_tables[k].to_markdown(floatfmt=".4f") + "\n")

    if ablation_tables:
        for k in ROI_METRIC_KEYS:
            csv_path = os.path.join(out_dir, f"ablation_{k}_{mode}.csv")
            ablation_tables[k].to_csv(csv_path)
            md_lines.append(f"\n## Ablation {k.upper()}\n\n")
            md_lines.append(ablation_tables[k].to_markdown(floatfmt=".4f") + "\n")

    if plasma_tables:
        for metric, tbl in plasma_tables.items():
            csv_path = os.path.join(out_dir, f"plasma_{metric}_{mode}.csv")
            tbl.to_csv(csv_path)
            md_lines.append(f"\n## Plasma × Region {metric.upper()} [SUVR, eq. 18]\n")
            md_lines.append(f"Subjects per bin: {plasma_counts[metric]}\n\n")
            md_lines.append(tbl.to_markdown(floatfmt=".4f") + "\n")

    md_path = os.path.join(out_dir, f"metrics_{mode}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Tables → {md_path}")


# ── checkpoint / dataset mapping ─────────────────────────────────────────────
_DIR_TO_SUBPATH = {
    "atrophy":    os.path.join("atrophy",  "72split"),
    "ptau217":    os.path.join("ptau217",  "72split"),
    "atrophy_v2": os.path.join("atrophy",  "80split"),
    "ptau217_v2": os.path.join("ptau217",  "80split"),
    "combined":   "atrophy_ptau217",
}


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="TauGenNet evaluation — canonical script")
    p.add_argument("--mode",           choices=["atrophy", "ptau217", "ptau217_biobert", "ptau217_mlp", "combined", "combined_mlp"], required=True)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--figures-dir",    type=str, default=None,
                   help="Subdirectory under results/figures/")
    p.add_argument("--records-dir",    type=str, default="results/records",
                   help="Directory for CSV/markdown tables")
    p.add_argument("--arch",           choices=["silu", "relu"], default="silu")
    p.add_argument("--use-mask",       action=argparse.BooleanOptionalAction, default=True,
                   help="Apply DK atlas mask to data (default: on; use --no-use-mask to disable)")
    p.add_argument("--n-steps",        type=int, default=500,
                   help="DDPM sampling steps (paper: 500)")
    p.add_argument("--skip-ablation",  action="store_true", default=False,
                   help="Skip ablation study (saves ~2× inference time)")
    p.add_argument("--normalized-supplementary", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Also emit the per-subject [0,1] NORMALIZED supplementary diagnostics "
                        "(M1n regional Pearson, M2 voxelwise cross-subject r, and the "
                        "normalized-vs-SUVR 2×2 summary). Off by default — all primary metrics "
                        "are SUVR; this adds the normalized counterparts for comparison only.")
    p.add_argument("--use-cached",     action=argparse.BooleanOptionalAction, default=None,
                   help="Load generated volumes from results/generated/<arch>/<mode> "
                        "instead of re-running DDPM inference. Default is context-aware: "
                        "cached UNLESS --checkpoint-dir is given (batch scripts that pass a "
                        "checkpoint do fresh inference; interactive runs use the cache). "
                        "Use --use-cached/--no-use-cached to force either way. Forces "
                        "--skip-ablation (the ablation needs fresh no-MRI inference).")
    p.add_argument("--rank-by", choices=["pearson","ssim","mae"], default="ssim",
                   help="Rank glass-brain best/median/worst by this per-subject metric.")
    p.add_argument("--glass-brain",    action=argparse.BooleanOptionalAction, default=True,
                   help="After metrics, render glass-brain/slice/scatter figures via "
                        "glass_brain_final.run_mode using the in-memory SUVR/normalized "
                        "volumes (no cache reload). Default: on; use --no-glass-brain to "
                        "skip. Requires nilearn.")
    p.add_argument("--use-mentor-split", action=argparse.BooleanOptionalAction, default=True,
                   help="Use Anil's fixed mentor split (default). Pass --no-use-mentor-split "
                        "to force the LEGACY random split — REQUIRED when evaluating cached "
                        "generations that were produced under the legacy split, or index-based "
                        "real↔gen pairing silently mismatches (applies to dataset_final/"
                        "combined/spatial only).")
    p.add_argument("--use-controls-322", action=argparse.BooleanOptionalAction, default=False,
                   help="Use the 322-subject CN+AD+MCI controls cohort (data/raw/controls_322/) "
                        "instead of the legacy AD/MCI split. Applies to dataset_final "
                        "('atrophy'/ptau modes) and dataset_combined ('combined'/'combined_mlp') "
                        "only — dataset_spatial has no controls_322 support yet.")
    p.add_argument("--suvr-as-cond",     action=argparse.BooleanOptionalAction, default=False,
                   help="Condition on regional SUVR (MLP) instead of/alongside the atrophy map. "
                        "Only applies to the dataset_final ('atrophy') branch.")
    p.add_argument("--generated-dir",  type=str, default=None,
                   help="Override the cached-generations directory used by --use-cached.")
    p.add_argument("--save-generated", action=argparse.BooleanOptionalAction, default=True,
                   help="Save generated volumes to results/generated/<arch>/<mode>/ "
                        "so glass_brain_final.py can reuse them with --use-cached.")
    p.add_argument("--unet-channels",  type=str, default="256,512,768",
                   help="UNet channel widths — must match training config (default: 256,512,768)")
    p.add_argument("--n-transformer",  type=int, default=3,
                   help="Transformer blocks per UNet level — must match training config (default: 3)")
    p.add_argument("--use-best",       action="store_true", default=False,
                   help="Load diff_{mode}_best.pt instead of diff_{mode}.pt")
    p.add_argument('--eval-on', choices=['test','val'], default='test')
    p.add_argument("--fold",           type=int, default=None,
                   help="CV fold index; evaluate on that fold's exact held-out test set.")
    p.add_argument("--n-folds",        type=int, default=5)
    p.add_argument("--phase6-manifest", type=str, default=None,
                   help="Pooled AD+MCI+CN manifest; phase6 test set via dataset_phase6.")
    p.add_argument("--dataset",        choices=["final", "combined", "spatial"], default=None,
                   help="Test-set dataset class. Default: infer from --mode (combined->combined, "
                        "else final). Use 'spatial' for train_spatial_atrophy models "
                        "(pair with --use-cached + --cond-mode).")
    p.add_argument("--cond-mode",      choices=["atrophy", "none", "ptau217"], default="atrophy",
                   help="For --dataset spatial: the cross-attention mode used at training, so "
                        "the test set matches the generated .npy ordering.")
    p.add_argument("--metrics-out",    type=str, default=None,
                   help="Write this fold's whole-brain DK86 test means (pearson/nrmse/ssim/"
                        "mse/mae + n) to a JSON for cross-fold CV aggregation.")
    args = p.parse_args()

    # Context-aware --use-cached default: when neither --use-cached nor --no-use-cached
    # is given, use the cache only if no checkpoint was specified. Passing a checkpoint
    # signals intent to run (and usually --save-generated) fresh DDPM inference.
    if args.use_cached is None:
        args.use_cached = args.checkpoint_dir is None
        print(f"--use-cached not set; defaulting to {args.use_cached} "
              f"({'no --checkpoint-dir → cached' if args.use_cached else '--checkpoint-dir given → fresh inference'}).")

    COND_MODE = args.mode

    # Checkpoint
    suffix = "_best" if args.use_best else ""
    if args.checkpoint_dir:
        CKPT_PATH = os.path.join(args.checkpoint_dir, f"diff_{COND_MODE}{suffix}.pt")
    else:
        CKPT_PATH = f"results/checkpoints/{COND_MODE}/diff_{COND_MODE}{suffix}.pt"

    # Figures directory
    if args.figures_dir:
        fig_subpath = args.figures_dir
    elif args.checkpoint_dir:
        key = os.path.basename(args.checkpoint_dir.rstrip("/"))
        fig_subpath = _DIR_TO_SUBPATH.get(key, key)
    else:
        fig_subpath = _DIR_TO_SUBPATH.get(COND_MODE, COND_MODE)

    FIG_DIR = os.path.join(FIGURES_DIR, fig_subpath)
    REC_DIR = args.records_dir
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Mode:     {COND_MODE}")
    print(f"Figures → {FIG_DIR}")
    print(f"Records → {REC_DIR}")

    # Dataset
    if args.arch == "relu":
        from src import dataset_v2_relu as _dataset_v2_relu
        from src import dataset_combined_relu as _dataset_combined_relu
        if COND_MODE == "combined":
            _, _val_ds, test_ds, *_ = _dataset_combined_relu.build_dataloaders(
                mode=COND_MODE, use_dk_mask=args.use_mask)
        else:
            _, _val_ds, test_ds, *_ = _dataset_v2_relu.build_dataloaders(
                mode=COND_MODE, use_dk_mask=args.use_mask)
    else:
        fold_kwargs = {} if args.fold is None else {"fold_idx": args.fold, "n_folds": args.n_folds}
        dataset_choice = args.dataset or ("combined" if COND_MODE in ("combined", "combined_mlp") else "final")
        if getattr(args, "phase6_manifest", None):
            from src import dataset_phase6 as _p6
            _p6cm = "combined" if args.cond_mode in ("combined", "combined_mlp") else args.cond_mode
            _, _val_ds, test_ds, *_ = _p6.build_dataloaders(
                manifest=args.phase6_manifest, use_dk_mask=args.use_mask,
                cond_mode=_p6cm, **fold_kwargs)
        elif dataset_choice == "spatial":
            from src import dataset_spatial as _dataset_spatial
            _, _val_ds, test_ds, *_ = _dataset_spatial.build_dataloaders(
                use_dk_mask=args.use_mask, cond_mode=args.cond_mode,
                use_mentor_split=args.use_mentor_split,
                use_controls_322=args.use_controls_322, **fold_kwargs)
        elif dataset_choice == "combined":
            _, _val_ds, test_ds, *_ = _dataset_combined.build_dataloaders(
                mode=COND_MODE, use_dk_mask=args.use_mask,
                use_mentor_split=args.use_mentor_split, use_controls_322=args.use_controls_322,
                **fold_kwargs)
        else:
            # ptau217_mlp / ptau217_biobert use the same data as ptau217
            # (1-dim scalar conditioning; only the text/MLP encoder differs)
            dataset_mode = "ptau217" if COND_MODE in ("ptau217_mlp", "ptau217_biobert") else COND_MODE
            _, _val_ds, test_ds, *_ = _dataset_v2.build_dataloaders(
                mode=dataset_mode, use_dk_mask=args.use_mask,
                use_mentor_split=args.use_mentor_split, use_controls_322=args.use_controls_322,
                suvr_as_cond=args.suvr_as_cond, **fold_kwargs)
        if args.eval_on == 'val': test_ds = _val_ds

    print(f"Test subjects: {len(test_ds)}")

    # Default cache dir when --generated-dir is not given. In CV mode each fold has a
    # different held-out test set, so folds MUST NOT share one flat dir (later folds
    # would clobber earlier .npy). Isolate by fold here as a safety net; callers should
    # still pass an explicit --generated-dir per variant.
    _default_gen_dir = os.path.join("results", "generated", args.arch, COND_MODE)
    if args.fold is not None:
        _default_gen_dir = os.path.join(_default_gen_dir, f"fold_{args.fold}")

    if args.use_cached:
        # Recompute metrics/figures from cached generations — no model, no GPU.
        if not args.skip_ablation:
            print("--use-cached: forcing --skip-ablation (ablation needs fresh inference).")
            args.skip_ablation = True
        ae = unet = conditioner = encode_cond = schedule = latent_std = None

        gen_dir = args.generated_dir or _default_gen_dir
        print(f"Loading cached generations from: {gen_dir}  (skipping DDPM inference)")
        real_norm, gen_masked, gen_raw, real_suvr, gen_suvr, plasma_vals = \
            load_cached_generations(test_ds, COND_MODE, gen_dir)
    else:
        # Models
        ch_list = tuple(int(x) for x in args.unet_channels.split(","))
        ae, unet, conditioner, latent_std, diff_losses = load_models(
            CKPT_PATH, COND_MODE, arch=args.arch,
            ch_list=ch_list, n_transformer=args.n_transformer)
        encode_cond = conditioner.encode
        schedule    = DiffusionSchedule()
        print(f"Checkpoint loaded: {CKPT_PATH}")

        # ── Inference (one pass, reused for all metrics and figures) ──────────────
        save_dir = None
        if args.save_generated:
            save_dir = args.generated_dir or _default_gen_dir
        real_norm, gen_masked, gen_raw, real_suvr, gen_suvr, plasma_vals = generate_all(
            ae, unet, schedule, encode_cond, latent_std, test_ds,
            cond_mode=COND_MODE, n_steps=args.n_steps, save_dir=save_dir,
        )

    # Shared color range for all brain slice figures
    pet_vmin, pet_vmax, err_vmin, err_vmax = compute_suvr_range(real_suvr, gen_suvr)
    print(f"\nSUVR display range: [{pet_vmin:.3f}, {pet_vmax:.3f}]  "
          f"Error range: [0, {err_vmax:.3f}]")

    # ── Whole-brain metrics ────────────────────────────────────────────────────
    # Single anatomical DK86 region mask (atlas labels 1-86) for every subject, so
    # metrics are not diluted by background or out-of-region voxels.
    dk_mask = _dk86_mask(test_ds)
    dk_bbox = _mask_bbox(dk_mask) if dk_mask is not None else None
    print("Metric region:", "DK86 atlas mask" if dk_mask is not None
          else "per-subject real>0 (no DK mask)")

    # All primary metrics are computed in SUVR (unnormalized) space. real_suvr is already
    # DK86-masked (background = 0); gen_suvr is unmasked, so apply the SAME support as
    # gen_masked here so real/generated cover identical voxels in SUVR space.
    gen_suvr_masked = [g * (dk_mask if dk_mask is not None else (r > 0))
                       for g, r in zip(gen_suvr, real_norm)]

    pearsons, nrmses, ssims, mses, maes = compute_wholebrain_metrics(
        real_suvr, gen_suvr_masked, brain_mask=dk_mask, bbox=dk_bbox)

    # ── Per-fold metrics JSON (whole-brain DK86 test means) for CV aggregation ──
    if args.metrics_out:
        import json as _json
        summary = {
            "pearson": float(np.nanmean(pearsons)), "nrmse": float(np.nanmean(nrmses)),
            "ssim":    float(np.nanmean(ssims)),    "mse":   float(np.nanmean(mses)),
            "mae":     float(np.nanmean(maes)),     "n":     int(len(real_norm)),
            "fold":    args.fold,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
        with open(args.metrics_out, "w") as _f:
            _json.dump(summary, _f, indent=2)
        print(f"Per-fold metrics JSON → {args.metrics_out}")

    # ── Canonical metric sets (pooled voxel / per-region / regional cross-subject) ──
    # All in SUVR (unnormalized) space, reusing the inference outputs above (no extra pass).
    cs1_mask  = dk_mask if dk_mask is not None else (np.stack(real_norm, axis=0) > 0).any(axis=0)
    set1_tbl  = compute_pooled_wholebrain_metrics(real_suvr, gen_suvr_masked, cs1_mask)
    label_vol = _dk86_label_volume(test_ds)
    set2_tbl = set3_tbl = Rreal86 = Rgen86 = None
    if label_vol is not None:
        set2_tbl              = compute_region_pooled_metrics(real_suvr, gen_suvr_masked, label_vol)
        set3_tbl, Rreal86, Rgen86 = compute_region_crosssubject_metrics(
            real_suvr, gen_suvr_masked, label_vol)
    else:
        print("WARNING: --no-use-mask → no DK atlas labels; skipping SET 2 & SET 3 "
              "(86-region metrics). SET 1 computed on union real>0 mask.")
    save_canonical_metric_sets(set1_tbl, set2_tbl, set3_tbl, Rreal86, Rgen86, COND_MODE, REC_DIR)

    # ── Figures for the 3 canonical R sets (saved alongside their CSVs) ────────
    plot_pooled_wholebrain_scatter(
        real_suvr, gen_suvr_masked, cs1_mask, set1_tbl, COND_MODE,
        save_path=os.path.join(FIG_DIR, f"pooled_wholebrain_scatter_{COND_MODE}.png"))
    if set2_tbl is not None:
        plot_region_pooled_r_bar(set2_tbl, COND_MODE,
            save_path=os.path.join(FIG_DIR, f"region_pooled_r_{COND_MODE}.png"))
    if set3_tbl is not None:
        plot_region_crosssubject_r_bar(set3_tbl, COND_MODE,
            save_path=os.path.join(FIG_DIR, f"region_crosssubject_r_{COND_MODE}.png"))

    # ── Figures: subject comparison + population ──────────────────────────────
    plot_subject_comparison(
        real_suvr, gen_suvr, pearsons, nrmses, ssims, mses, maes,
        COND_MODE, pet_vmin, pet_vmax, err_vmax,
        save_path=os.path.join(FIG_DIR, f"subject_comparison_{COND_MODE}.png"),
    )
    plot_population_comparison(
        real_suvr, gen_suvr, COND_MODE, pet_vmin, pet_vmax, err_vmax,
        save_path=os.path.join(FIG_DIR, f"population_comparison_{COND_MODE}.png"),
    )
    plot_wholebrain_boxplot(
        pearsons, nrmses, ssims, mses, maes, COND_MODE,
        save_path=os.path.join(FIG_DIR, f"wholebrain_boxplot_{COND_MODE}.png"),
    )

    # ── Cross-subject voxelwise Pearson r (SUVR — primary) ─────────────────────
    print("\nComputing cross-subject voxelwise Pearson r…")
    # DK86 mask when available, else union of real>0. SUVR keeps the absolute tau burden
    # that the cross-subject correlation is meant to measure.
    cs_brain_mask = dk_mask if dk_mask is not None else (np.stack(real_norm, axis=0) > 0).any(axis=0)
    cs_r_map = compute_crosssubject_voxelwise_r(real_suvr, gen_suvr_masked,
                                                brain_mask=cs_brain_mask, label=" [SUVR]")
    cs_roi_r = compute_crosssubject_roi_r(cs_r_map)
    cs_mean_r = float(cs_r_map[cs_brain_mask].mean())
    plot_crosssubject_r_map(cs_r_map, real_norm, COND_MODE,
        save_path=os.path.join(FIG_DIR, f"crosssubject_r_map_{COND_MODE}.png"))
    plot_crosssubject_roi_r(cs_roi_r, COND_MODE,
        save_path=os.path.join(FIG_DIR, f"crosssubject_roi_r_{COND_MODE}.png"))

    # ── Supplementary normalized [0,1] voxelwise cross-subject r (opt-in) ──────
    # Same voxels as the SUVR map; only intensity space differs (per-subject min-max
    # normalization removes the between-subject tau burden). Off unless requested.
    crosssubject_summary = None
    if args.normalized_supplementary:
        cs_r_map_norm = compute_crosssubject_voxelwise_r(
            real_norm, gen_masked, brain_mask=cs_brain_mask,
            label=" [normalized / burden-removed]")
        cs_roi_r_norm = compute_crosssubject_roi_r(cs_r_map_norm)
        cs_mean_r_norm = float(cs_r_map_norm[cs_brain_mask].mean())
        plot_crosssubject_r_map(cs_r_map_norm, real_norm, f"{COND_MODE} (normalized)",
            save_path=os.path.join(FIG_DIR, f"crosssubject_r_map_{COND_MODE}_normalized.png"))
        plot_crosssubject_roi_r(cs_roi_r_norm, f"{COND_MODE} (normalized)",
            save_path=os.path.join(FIG_DIR, f"crosssubject_roi_r_{COND_MODE}_normalized.png"))
        crosssubject_summary = {
            "SUVR":       {"_mean": cs_mean_r,      **cs_roi_r},
            "normalized": {"_mean": cs_mean_r_norm, **cs_roi_r_norm},
        }

    # ── ROI per-subject metrics ────────────────────────────────────────────────
    roi_tables, roi_raw = compute_roi_metrics(real_suvr, gen_suvr_masked)
    plot_roi_metrics_boxplot(
        roi_raw, COND_MODE,
        save_path=os.path.join(FIG_DIR, f"roi_metrics_boxplot_{COND_MODE}.png"),
    )

    # ── Ablation study ─────────────────────────────────────────────────────────
    ablation_tables = None
    if not args.skip_ablation:
        ablation_tables = compute_ablation_metrics(
            ae, unet, schedule, encode_cond, latent_std, test_ds, n_steps=args.n_steps)
        plot_ablation_comparison(
            ablation_tables, COND_MODE,
            save_path=os.path.join(FIG_DIR, f"ablation_{COND_MODE}.png"),
        )

    # ── Regional-mean SUVR (6-ROI cross-subject, eq.18) — DISABLED per user request ──
    # (the 86-region SET 3 in save_canonical_metric_sets is the one we report; this
    #  6-ROI "Regional-mean SUVR" table is intentionally turned off.)
    # regional_tables, Rreal, Rgen = compute_regional_overall(real_suvr, gen_suvr)
    # regional_tables["ssim"] = roi_tables["ssim"]
    # plot_regional_overall_heatmap(regional_tables, COND_MODE, FIG_DIR)
    regional_tables, Rreal, Rgen = None, None, None   # plasma table recomputes Rreal/Rgen if needed

    # ── M1n: regional-mean cross-subject Pearson in NORMALIZED space (opt-in) ──
    regional_norm_table = None
    if args.normalized_supplementary:
        regional_norm_table = compute_regional_overall_normalized(real_norm, gen_raw)

    # ── Plasma × region — ptau217 / combined only (stratified by plasma bin) ──
    plasma_tables, plasma_counts = None, None
    if plasma_vals is not None:
        plasma_arr = np.asarray(plasma_vals)
        print(f"\nPlasma p-tau217 test range: {plasma_arr.min():.2f} – {plasma_arr.max():.2f}")
        plasma_tables, plasma_counts = compute_plasma_tables(
            real_suvr, gen_suvr, plasma_vals, Rreal=Rreal, Rgen=Rgen)
        for metric in ("nrmse", "mse", "mae", "pearson"):
            plot_plasma_heatmap(
                plasma_tables[metric], metric, COND_MODE,
                save_path=os.path.join(FIG_DIR, f"plasma_{metric}_{COND_MODE}.png"),
            )
    else:
        print("\nSkipping plasma × region tables (atrophy mode — no plasma p-tau217).")

    # ── Save all tables ────────────────────────────────────────────────────────
    save_tables_to_disk(roi_tables, ablation_tables, plasma_tables, plasma_counts,
                        regional_tables, COND_MODE, REC_DIR,
                        regional_norm_table=regional_norm_table,
                        crosssubject_summary=crosssubject_summary)

    # ── Glass-brain / slice / scatter figures (opt-in) ───────────────────────────
    # Reuses the canonical renderer (scripts/glass_brain_final.py) but skips its cache
    # reload by passing the in-memory volumes. Import is deferred so the heavy nilearn
    # dependency only loads when --glass-brain is set.
    if args.glass_brain:
        print("\nRendering glass-brain figures (glass_brain_final.run_mode)…")
        try:
            from scripts.glass_brain_final import run_mode as _gb_run_mode
        except ImportError:
            from glass_brain_final import run_mode as _gb_run_mode
        # glass_brain_final masks generated SUVR by the per-subject real>0 support for display.
        gb_gen_suvr = [g * (r > 0) for g, r in zip(gen_suvr, real_norm)]
        gb_dir = os.path.join(FIG_DIR, "glass_brain")
        os.makedirs(gb_dir, exist_ok=True)
        _gb_run_mode(COND_MODE, args.arch, gb_dir, use_dk_mask=args.use_mask,
                     data=(real_suvr, gb_gen_suvr, real_norm, gen_masked, test_ds),
                     rank_by=args.rank_by)

    print("\nDone.")
