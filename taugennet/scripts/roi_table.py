"""
scripts/roi_table.py

Per-ROI NRMSE and 3D SSIM for TauGenNet SiLU runs, plus two box plots.

Outputs:
  - TABLE III printed to stdout: per-ROI NRMSE and SSIM (mean±std) for 6 regions × 3 modes
  - results/figures/silu/boxplot_nrmse_ssim.png
      Whole-brain per-subject NRMSE and SSIM: 3 boxes (one per conditioning mode)
  - results/figures/silu/boxplot_roi_nrmse_ssim.png
      Per-ROI grouped: 6 region clusters × 3 boxes per cluster (one per mode), 2 rows (NRMSE / SSIM)
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim3d
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEVICE, VOL_SHAPE
from src import dataset_final       as _dataset_v2
from src import dataset_combined as _dataset_combined
from src.diffusion import DiffusionSchedule
from src.inference import load_models, synthesize_tau_pet, synthesize_no_mri

# ── Harvard-Oxford atlas ROI definitions ─────────────────────────────────────
# All data is in MNI152 space, so one atlas covers all subjects.
# Cortical atlas labels (cort-maxprob-thr25-1mm):
#   9=Temporal Pole, 15/16/17=Inf Temporal Gyrus ant/post/temporocc,
#   31=Cingulate Gyrus posterior, 35/36=Parahippocampal ant/post,
#   38/39/40=Temporal Fusiform ant/post + Temporal Occipital Fusiform
# Subcortical atlas labels (sub-maxprob-thr25-1mm):
#   10=L Hippocampus, 20=R Hippocampus
ATLAS_ROIS = {
    "Entorhinal":      ("cort", [9]),
    "Parahippocampal": ("cort", [35, 36]),
    "Hippocampus":     ("sub",  [10, 20]),
    "Fusiform":        ("cort", [38, 39, 40]),
    "Inf. Temporal":   ("cort", [15, 16, 17]),
    "Post. Cingulate": ("cort", [31]),
}
ROI_ORDER = ["Entorhinal", "Parahippocampal", "Hippocampus",
             "Fusiform", "Inf. Temporal", "Post. Cingulate"]

_ATLAS_MASK_CACHE = {}


def build_atlas_masks(vol_shape):
    """Precompute boolean ROI masks from Harvard-Oxford atlas resampled to vol_shape."""
    if vol_shape in _ATLAS_MASK_CACHE:
        return _ATLAS_MASK_CACHE[vol_shape]

    from nilearn import datasets
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
    ho_sub  = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')

    def _resample(nii_img):
        t = torch.from_numpy(nii_img.get_fdata().astype(np.float32))
        t = t.unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=vol_shape, mode="nearest")
        return t.squeeze().numpy().astype(np.int16)

    dat_c = _resample(ho_cort.maps)
    dat_s = _resample(ho_sub.maps)

    masks = {}
    for rname, (src, label_ids) in ATLAS_ROIS.items():
        dat = dat_c if src == "cort" else dat_s
        mask = np.zeros(vol_shape, dtype=bool)
        for lid in label_ids:
            mask |= (dat == lid)
        masks[rname] = mask
        print(f"  atlas mask {rname}: {mask.sum()} voxels")

    _ATLAS_MASK_CACHE[vol_shape] = masks
    return masks

COLORS = ["#4C72B0", "#DD8452", "#55A868"]  # blue / orange / green

CONFIGS = [
    dict(mode="atrophy",  label="Atrophy",
         ckpt_dir="results/checkpoints/silu/atrophy",  dataset="v2"),
    dict(mode="ptau217",  label="p-tau217",
         ckpt_dir="results/checkpoints/silu/ptau217",  dataset="v2"),
    dict(mode="combined", label="Combined",
         ckpt_dir="results/checkpoints/silu/combined", dataset="combined"),
]

WB_BOXPLOT_OUT  = "results/figures/silu/boxplot_nrmse_ssim.png"
ROI_BOXPLOT_OUT = "results/figures/silu/boxplot_roi_nrmse_ssim.png"


# ── ROI metric helpers ────────────────────────────────────────────────────────

def roi_nrmse(real, gen, mask):
    r, g = real[mask], gen[mask]
    return float(np.sqrt(np.mean((g - r) ** 2)) / (r.max() - r.min() + 1e-8))


def roi_ssim(real, gen, mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return float("nan")
    mn, mx = coords.min(0), coords.max(0)
    r = real[mn[0]:mx[0]+1, mn[1]:mx[1]+1, mn[2]:mx[2]+1]
    g = gen [mn[0]:mx[0]+1, mn[1]:mx[1]+1, mn[2]:mx[2]+1]
    min_dim  = min(r.shape)
    win_size = min(7, min_dim) if min(min_dim, 7) % 2 == 1 else min(7, min_dim) - 1
    win_size = max(win_size, 3)
    return float(ssim3d(r, g, data_range=1.0, win_size=win_size))


def whole_brain_nrmse(real, gen):
    return float(np.sqrt(np.mean((gen - real) ** 2)) / (real.max() - real.min() + 1e-8))


def whole_brain_ssim(real, gen):
    return float(ssim3d(real, gen, data_range=1.0))


def compute_all_metrics(real, gen):
    diff  = gen - real
    mse   = float(np.mean(diff ** 2))
    mae   = float(np.mean(np.abs(diff)))
    nrmse = float(np.sqrt(mse) / (real.max() - real.min() + 1e-8))
    ssim_score = float(ssim3d(real, gen, data_range=1.0))
    r, _  = pearsonr(real.ravel(), gen.ravel())
    return {"nrmse": nrmse, "mae": mae, "mse": mse, "pearson": float(r), "ssim": ssim_score}


# ── Per-mode evaluation ───────────────────────────────────────────────────────

def evaluate_mode(cfg, volumes_dir=None):
    print(f"\n{'='*64}")
    print(f"  {cfg['label']}")
    print(f"{'='*64}")

    ckpt_path = os.path.join(cfg["ckpt_dir"], f"diff_{cfg['mode']}_best.pt")
    ae, unet, conditioner, latent_std, _ = load_models(
        ckpt_path, cfg["mode"], device=DEVICE, arch="silu"
    )
    schedule    = DiffusionSchedule(device=DEVICE)
    encode_cond = conditioner.encode

    if cfg["dataset"] == "combined":
        _, _, test_ds, _, _, _ = _dataset_combined.build_dataloaders(
            mode="combined", batch_size=1, use_dk_mask=True
        )
    else:
        _, _, test_ds, _, _, _ = _dataset_final.build_dataloaders(
            mode=cfg["mode"], batch_size=1, use_dk_mask=True
        )

    print(f"Test subjects: {len(test_ds)}")
    roi_masks = build_atlas_masks(VOL_SHAPE)

    nrmse_roi = {r: [] for r in ROI_ORDER}
    ssim_roi  = {r: [] for r in ROI_ORDER}
    wb_nrmses, wb_ssims = [], []

    for i in range(len(test_ds)):
        pet, mri, cond = test_ds[i]
        real_np = pet.squeeze().numpy()

        cache_path = None
        if volumes_dir is not None:
            cache_path = os.path.join(volumes_dir, cfg["mode"], f"subject_{i:03d}.npy")

        if cache_path and os.path.exists(cache_path):
            gen_np = np.load(cache_path)
            if (i + 1) % 10 == 0 or (i + 1) == len(test_ds):
                print(f"  subject {i+1}/{len(test_ds)}  [cached]")
        else:
            gen_np = synthesize_tau_pet(
                mri.unsqueeze(0), cond, ae, unet, schedule, encode_cond,
                latent_std, device=DEVICE, n_steps=500, sampler="ddpm"
            ).squeeze().numpy()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, gen_np)

        wb_nrmses.append(whole_brain_nrmse(real_np, gen_np))
        wb_ssims.append(whole_brain_ssim(real_np, gen_np))

        for rname in ROI_ORDER:
            mask = roi_masks[rname]
            nrmse_roi[rname].append(roi_nrmse(real_np, gen_np, mask))
            ssim_roi [rname].append(roi_ssim (real_np, gen_np, mask))

        if cache_path is None and ((i + 1) % 10 == 0 or (i + 1) == len(test_ds)):
            print(f"  subject {i+1}/{len(test_ds)}  "
                  f"wb_nrmse={wb_nrmses[-1]:.4f}  wb_ssim={wb_ssims[-1]:.4f}")

    nrmse_stats = {r: (np.nanmean(nrmse_roi[r]), np.nanstd(nrmse_roi[r])) for r in ROI_ORDER}
    ssim_stats  = {r: (np.nanmean(ssim_roi [r]), np.nanstd(ssim_roi [r])) for r in ROI_ORDER}

    return nrmse_stats, ssim_stats, wb_nrmses, wb_ssims, nrmse_roi, ssim_roi


# ── Ablation evaluation (atrophy mode only) ───────────────────────────────────

def evaluate_ablation(volumes_dir=None):
    """Per-subject NRMSE and SSIM: MRI+Atrophy vs. Atrophy-only (zeroed MRI latent)."""
    print(f"\n{'='*64}")
    print(f"  Ablation: MRI+Atrophy vs. Atrophy-only")
    print(f"{'='*64}")
    cfg = CONFIGS[0]  # atrophy mode
    ckpt_path = os.path.join(cfg["ckpt_dir"], f"diff_{cfg['mode']}_best.pt")
    ae, unet, conditioner, latent_std, _ = load_models(
        ckpt_path, cfg["mode"], device=DEVICE, arch="silu"
    )
    schedule    = DiffusionSchedule(device=DEVICE)
    encode_cond = conditioner.encode

    _, _, test_ds, _, _, _ = _dataset_final.build_dataloaders(
        mode="atrophy", batch_size=1, use_dk_mask=True
    )
    print(f"Test subjects: {len(test_ds)}")

    METRIC_KEYS = ["nrmse", "mae", "mse", "pearson", "ssim"]
    with_mri = {k: [] for k in METRIC_KEYS}
    no_mri   = {k: [] for k in METRIC_KEYS}

    for i in range(len(test_ds)):
        pet, mri, cond = test_ds[i]
        real_np = pet.squeeze().numpy()

        # MRI+Atrophy — use cache from main evaluate_mode run
        cache_with = None
        cache_no   = None
        if volumes_dir is not None:
            cache_with = os.path.join(volumes_dir, "atrophy",         f"subject_{i:03d}.npy")
            cache_no   = os.path.join(volumes_dir, "atrophy_no_mri",  f"subject_{i:03d}.npy")

        if cache_with and os.path.exists(cache_with):
            gen_with = np.load(cache_with)
        else:
            gen_with = synthesize_tau_pet(
                mri.unsqueeze(0), cond, ae, unet, schedule, encode_cond,
                latent_std, device=DEVICE, n_steps=500, sampler="ddpm"
            ).squeeze().numpy()
            if cache_with:
                os.makedirs(os.path.dirname(cache_with), exist_ok=True)
                np.save(cache_with, gen_with)

        if cache_no and os.path.exists(cache_no):
            gen_no = np.load(cache_no)
        else:
            gen_no = synthesize_no_mri(
                mri.unsqueeze(0), cond, ae, unet, schedule, encode_cond,
                latent_std, device=DEVICE, n_steps=500
            ).squeeze().numpy()
            if cache_no:
                os.makedirs(os.path.dirname(cache_no), exist_ok=True)
                np.save(cache_no, gen_no)

        m_with = compute_all_metrics(real_np, gen_with)
        m_no   = compute_all_metrics(real_np, gen_no)
        for k in METRIC_KEYS:
            with_mri[k].append(m_with[k])
            no_mri[k].append(m_no[k])

        if (i + 1) % 10 == 0 or (i + 1) == len(test_ds):
            print(f"  subject {i+1}/{len(test_ds)}  "
                  f"w/MRI nrmse={m_with['nrmse']:.4f}  no-MRI nrmse={m_no['nrmse']:.4f}")

    return with_mri, no_mri


def print_ablation_table(with_mri, no_mri):
    METRIC_DISPLAY = [
        ("nrmse",   "NRMSE",      "lower"),
        ("mae",     "MAE",        "lower"),
        ("mse",     "MSE",        "lower"),
        ("pearson", "Pearson R",  "higher"),
        ("ssim",    "3D SSIM",    "higher"),
    ]
    w = 14
    sep = "=" * (2 + w + 28 + 28)
    print(f"\n{sep}")
    print("ABLATION: MRI+Atrophy vs. Atrophy-only  (whole-brain, per-subject)")
    print(sep)
    print(f"{'Metric':<{w}}  {'MRI + Atrophy':>26}  {'Atrophy only':>26}")
    print("-" * (2 + w + 28 + 28))
    for key, label, direction in METRIC_DISPLAY:
        wm = np.array(with_mri[key])
        nm = np.array(no_mri[key])
        print(f"{label:<{w}}  {wm.mean():>12.4f} ± {wm.std():>8.4f}  "
              f"{nm.mean():>12.4f} ± {nm.std():>8.4f}  ({direction} is better)")
    print(sep)


def make_ablation_boxplot(with_mri, no_mri):
    out_path = "results/figures/silu/boxplot_ablation.png"
    labels   = ["MRI + Atrophy", "Atrophy only"]
    colors   = ["#4C72B0", "#DD8452"]
    rng      = np.random.default_rng(42)

    PANELS = [
        ("nrmse",   "NRMSE",     "Whole-Brain NRMSE\n(lower is better)"),
        ("mae",     "MAE",       "Whole-Brain MAE\n(lower is better)"),
        ("mse",     "MSE",       "Whole-Brain MSE\n(lower is better)"),
        ("pearson", "Pearson R", "Whole-Brain Pearson R\n(higher is better)"),
        ("ssim",    "3D SSIM",   "Whole-Brain 3D SSIM\n(higher is better)"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("TauGenNet (SiLU, Atrophy) — MRI Ablation", fontsize=13, fontweight="bold")

    for ax, (key, ylabel, title) in zip(axes, PANELS):
        data_pair = [with_mri[key], no_mri[key]]
        bp = ax.boxplot(data_pair, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for k, (d, color) in enumerate(zip(data_pair, colors), start=1):
            jitter = rng.uniform(-0.15, 0.15, size=len(d))
            ax.scatter(k + jitter, d, color=color, alpha=0.5, s=18, zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation box plot saved -> {out_path}")


# ── Whole-brain box plot ──────────────────────────────────────────────────────

def make_wb_boxplot(all_wb):
    """3 boxes per metric (one per mode), whole-brain per-subject distribution."""
    labels     = [x[0] for x in all_wb]
    nrmse_data = [x[1] for x in all_wb]
    ssim_data  = [x[2] for x in all_wb]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("TauGenNet (SiLU) — Per-Subject Whole-Brain Distribution",
                 fontsize=13, fontweight="bold")

    for ax, data, ylabel, title in [
        (axes[0], nrmse_data, "NRMSE",   "Whole-Brain NRMSE (lower is better)"),
        (axes[1], ssim_data,  "3D SSIM", "Whole-Brain 3D SSIM (higher is better)"),
    ]:
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for i, (d, color) in enumerate(zip(data, COLORS), start=1):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(d))
            ax.scatter(i + jitter, d, color=color, alpha=0.5, s=18, zorder=3)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(WB_BOXPLOT_OUT), exist_ok=True)
    plt.savefig(WB_BOXPLOT_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Whole-brain box plot saved -> {WB_BOXPLOT_OUT}")


# ── Per-ROI grouped box plot ──────────────────────────────────────────────────

def make_roi_boxplot(all_roi):
    """
    2-row figure (NRMSE top, SSIM bottom).
    6 grouped clusters along x-axis (one per ROI).
    3 boxes per cluster (one per conditioning mode).
    """
    labels = [x[0] for x in all_roi]   # ["Atrophy", "p-tau217", "Combined"]
    n_modes = len(labels)
    n_rois  = len(ROI_ORDER)
    width   = 0.22
    rng     = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.suptitle("TauGenNet (SiLU) — Per-ROI Distribution Across Conditioning Modes",
                 fontsize=13, fontweight="bold")

    for row, (_, ylabel, title, get_data) in enumerate([
        ("nrmse", "NRMSE",   "Per-ROI NRMSE (lower is better)",
         lambda entry: entry[1]),
        ("ssim",  "3D SSIM", "Per-ROI 3D SSIM (higher is better)",
         lambda entry: entry[2]),
    ]):
        ax = axes[row]
        group_centers = np.arange(n_rois)

        for m_idx, (entry, color) in enumerate(zip(all_roi, COLORS)):
            roi_data = get_data(entry)   # dict {roi_name: [per-subject values]}
            offsets  = (np.arange(n_modes) - (n_modes - 1) / 2) * width
            x_pos    = [group_centers[r] + offsets[m_idx] for r in range(n_rois)]
            data     = [roi_data[rname] for rname in ROI_ORDER]

            bp = ax.boxplot(data, positions=x_pos, widths=width * 0.85,
                            patch_artist=True, notch=False, manage_ticks=False,
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2),
                            flierprops=dict(marker=".", markersize=3, alpha=0.4))
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # jittered individual points
            for r_idx, d in enumerate(data):
                jitter = rng.uniform(-width * 0.3, width * 0.3, size=len(d))
                ax.scatter(x_pos[r_idx] + jitter, d,
                           color=color, alpha=0.45, s=10, zorder=3)

        ax.set_xticks(group_centers)
        ax.set_xticklabels(ROI_ORDER, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if row == 0:
            handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.7)
                       for c in COLORS]
            ax.legend(handles, labels, loc="upper right", fontsize=9,
                      framealpha=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(ROI_BOXPLOT_OUT), exist_ok=True)
    plt.savefig(ROI_BOXPLOT_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-ROI box plot saved   -> {ROI_BOXPLOT_OUT}")


# ── Whole-brain summary table ─────────────────────────────────────────────────

def print_wb_table(all_wb):
    w = 60
    print("\n")
    print("=" * w)
    print("WHOLE-BRAIN NRMSE AND 3D SSIM — PER CONDITIONING MODE")
    print("=" * w)
    print(f"{'Mode':<14}  {'NRMSE mean±std':>20}  {'SSIM mean±std':>20}")
    print("-" * w)
    for label, nrmses, ssims in all_wb:
        print(f"{label:<14}  "
              f"{np.mean(nrmses):>8.4f} ± {np.std(nrmses):<8.4f}  "
              f"{np.mean(ssims):>8.4f} ± {np.std(ssims):.4f}")
    print("=" * w)


# ── Per-ROI table printer ──────────────────────────────────────────────────────

def print_table(results):
    col_w   = 17
    label_w = 24
    divider = "-" * (label_w + col_w * len(ROI_ORDER))

    print("\n")
    print("=" * (label_w + col_w * len(ROI_ORDER)))
    print("TABLE III")
    print("COMPARISON EXPERIMENT OF NRMSE AND 3D SSIM FOR TAUGENNET")
    print("ACROSS DIFFERENT BRAIN REGIONS")
    print("=" * (label_w + col_w * len(ROI_ORDER)))
    print(f"{'Method':<{label_w}}" + "".join(f"{r:>{col_w}}" for r in ROI_ORDER))
    print(divider)

    for cfg, (nrmse_d, ssim_d, *_) in results:
        print(f"\n{cfg['label']}")
        nrow = f"  {'NRMSE (mean+/-std)':<{label_w-2}}" + \
               "".join(f"{nrmse_d[r][0]:>{col_w-6}.4f}+/-{nrmse_d[r][1]:.3f} "
                       for r in ROI_ORDER)
        srow = f"  {'SSIM  (mean+/-std)':<{label_w-2}}" + \
               "".join(f"{ssim_d[r][0]:>{col_w-6}.4f}+/-{ssim_d[r][1]:.3f} "
                       for r in ROI_ORDER)
        print(nrow)
        print(srow)
        print(divider)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes-dir", default=None,
                        help="Directory to cache/load synthesized volumes (skips GPU synthesis on re-runs)")
    args = parser.parse_args()

    results = []
    all_wb  = []
    all_roi = []

    for cfg in CONFIGS:
        nrmse_d, ssim_d, wb_nrmses, wb_ssims, nrmse_roi, ssim_roi = evaluate_mode(
            cfg, volumes_dir=args.volumes_dir
        )
        results.append((cfg, (nrmse_d, ssim_d, wb_nrmses, wb_ssims)))
        all_wb.append( (cfg["label"], wb_nrmses, wb_ssims))
        all_roi.append((cfg["label"], nrmse_roi, ssim_roi))

    print_wb_table(all_wb)
    print_table(results)
    make_wb_boxplot(all_wb)
    make_roi_boxplot(all_roi)

    with_mri, no_mri = evaluate_ablation(volumes_dir=args.volumes_dir)
    print_ablation_table(with_mri, no_mri)
    make_ablation_boxplot(with_mri, no_mri)

    print("\nDone.")
