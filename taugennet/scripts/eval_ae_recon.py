#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Evaluate autoencoder reconstruction quality on the test set.

Loads the AE from a checkpoint, runs every test subject through it, then
produces a figure showing best / median / worst reconstruction by per-subject
MSE — three rows (subjects) × three columns (real | recon | error), one axial
slice per panel plus coronal and sagittal insets.

Usage
-----
python scripts/eval_ae_recon.py
python scripts/eval_ae_recon.py --checkpoint results/checkpoints/taugennet_checkpoint_paper.pt
python scripts/eval_ae_recon.py --mode ptau217 --figures-dir results/figures/ae_recon
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

from src.config import DEVICE, VOL_SHAPE, FIGURES_DIR, AE_CHECKPOINT_PATH, LATENT_CH
from src.models import Autoencoder3D
from src import dataset_final as _dataset


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_ssim(r, g, data_range=1.0):
    """Windowed 3-D SSIM with an odd window that fits the smallest axis."""
    win = min(7, min(r.shape))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return float("nan")
    return float(ssim(r, g, data_range=data_range, win_size=win))


def _dk86_mask_and_bbox(ds):
    """Return (mask, bbox) for the DK86 region union (atlas labels 1-86).

    mask: (H,W,D) bool — the dataset's shared DK mask (atlas_data > 0).
    bbox: tuple of slices = bounding box of the mask, for windowed SSIM.
    Falls back to None if the dataset was built with --no-use-mask.
    """
    dk = getattr(ds, "_dk_mask", None)
    if dk is None:
        return None, None
    mask = (dk.squeeze(0).numpy() > 0)
    nz = np.argwhere(mask)
    lo, hi = nz.min(0), nz.max(0) + 1
    bbox = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return mask, bbox


def _masked_metrics(real, recon, mask, bbox):
    """All 5 metrics restricted to the DK86 mask (real/recon in [0,1] space).

    Voxel metrics (MSE/MAE/NRMSE/Pearson) are computed over mask voxels only;
    SSIM is windowed so it is computed on the mask's bounding-box crop. When
    mask is None (no DK mask), falls back to per-subject real>0 support.
    """
    if mask is None:
        mask = real > 0
        nz = np.argwhere(mask)
        lo, hi = nz.min(0), nz.max(0) + 1
        bbox = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    rv, gv = real[mask], recon[mask]
    diff   = gv - rv
    mse    = float(np.mean(diff ** 2))
    mae    = float(np.mean(np.abs(diff)))
    nrmse  = float(np.sqrt(mse) / (rv.max() - rv.min() + 1e-8))
    pear   = (float("nan") if rv.std() < 1e-8 or gv.std() < 1e-8
              else float(pearsonr(rv, gv)[0]))
    ssim_s = _safe_ssim(real[bbox], recon[bbox])
    return {"mse": mse, "mae": mae, "nrmse": nrmse, "pearson": pear, "ssim": ssim_s}

def load_ae(checkpoint_path: str, latent_ch: int = None) -> Autoencoder3D:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    # Prefer an explicit override (needed for older AE checkpoints that didn't save
    # latent_ch); else read it from the checkpoint; else fall back to the config default.
    latent_ch = latent_ch or ckpt.get("latent_ch", LATENT_CH)
    ae = Autoencoder3D(latent_ch=latent_ch).to(DEVICE)
    state = ckpt["ae"] if "ae" in ckpt else ckpt
    ae.load_state_dict(state)
    ae.eval()
    print(f"AE latent_ch={latent_ch}")
    return ae


@torch.no_grad()
def reconstruct(ae: Autoencoder3D, vol: torch.Tensor) -> torch.Tensor:
    """vol: (1,H,W,D) on DEVICE, normalised [0,1].  Returns recon same shape."""
    vol = vol.unsqueeze(0)          # → (1,1,H,W,D)
    recon, _, _ = ae(vol)
    return recon.squeeze(0).clamp(0, 1)   # → (1,H,W,D)


def mid(vol: np.ndarray, axis: int) -> np.ndarray:
    """Central 2-D slice along axis from a (H,W,D) array."""
    idx = vol.shape[axis] // 2
    return np.take(vol, idx, axis=axis)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default=AE_CHECKPOINT_PATH)
    p.add_argument("--mode",         default="atrophy",
                   help="Dataset mode (atrophy|ptau217) — combined not needed for AE eval")
    p.add_argument("--figures-dir",  default=None,
                   help="Subpath under results/figures/ (default: ae_recon)")
    p.add_argument("--split",        default="test",
                   choices=["test", "val", "train"],
                   help="Which split to evaluate (default: test)")
    p.add_argument("--use-mask",     action=argparse.BooleanOptionalAction, default=True,
                   help="Apply DK atlas mask (default: on; use --no-use-mask to disable)")
    p.add_argument("--fold",         type=int, default=None,
                   help="CV fold index (0-based). When set, the split matches the "
                        "fold's train/val/test partition — use --split val for selection.")
    p.add_argument("--n-folds",      type=int, default=5)
    p.add_argument("--latent-ch",    type=int, default=None,
                   help="Override the AE latent channels. Needed for older checkpoints "
                        "that did not save 'latent_ch' (e.g. grid-search lch4/6/8). When "
                        "omitted, read from the checkpoint, falling back to the config default.")
    p.add_argument("--metrics-out",  default=None,
                   help="If set, write {mse,mae,nrmse,pearson,ssim,n} (subject means) "
                        "to this JSON path — for automated sweeps (run_grid_search step0).")
    p.add_argument("--no-figures",   action="store_true",
                   help="Skip figure/histogram generation (faster for sweeps).")
    args = p.parse_args()

    fig_subdir = args.figures_dir or "ae_recon"
    fig_dir = os.path.join(FIGURES_DIR, fig_subdir)
    if not args.no_figures:
        os.makedirs(fig_dir, exist_ok=True)

    # ── dataset ───────────────────────────────────────────────────────────────
    fold_kwargs = {} if args.fold is None else {"fold_idx": args.fold, "n_folds": args.n_folds}
    train_ds, val_ds, test_ds, *_ = _dataset.build_dataloaders(
        mode=args.mode, use_dk_mask=args.use_mask, **fold_kwargs)
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds = split_map[args.split]
    fold_str = "" if args.fold is None else f" (fold {args.fold}/{args.n_folds})"
    print(f"Evaluating AE on {args.split} split{fold_str}: {len(ds)} subjects")

    # ── load AE ───────────────────────────────────────────────────────────────
    ae = load_ae(args.checkpoint, latent_ch=args.latent_ch)
    print(f"AE loaded from: {args.checkpoint}")

    # DK86 region mask (atlas labels 1-86) — metrics are computed inside this only,
    # so background/out-of-region voxels never dilute the score.
    dk_mask, dk_bbox = _dk86_mask_and_bbox(ds)
    print("Metric region:", "DK86 atlas mask" if dk_mask is not None
          else "per-subject real>0 (no DK mask — used --no-use-mask)")

    # ── run inference ─────────────────────────────────────────────────────────
    METRIC_KEYS = ["mse", "mae", "nrmse", "pearson", "ssim"]
    metrics = {k: [] for k in METRIC_KEYS}
    reals, recons = [], []
    for i in range(len(ds)):
        pet, _, _ = ds[i]                       # (1,H,W,D)
        pet = pet.to(DEVICE)
        rec = reconstruct(ae, pet)
        real_np = pet.squeeze(0).cpu().numpy()   # (H,W,D)
        rec_np  = rec.squeeze(0).cpu().numpy()
        m = _masked_metrics(real_np, rec_np, dk_mask, dk_bbox)
        for k in METRIC_KEYS:
            metrics[k].append(m[k])
        reals.append(real_np)
        recons.append(rec_np)

    metrics = {k: np.array(v) for k, v in metrics.items()}
    mses = metrics["mse"]
    order = np.argsort(mses)
    best_i   = order[0]
    median_i = order[len(order) // 2]
    worst_i  = order[-1]

    print(f"\nPer-subject metrics (in-DK86, mean ± std over n={len(ds)}):")
    for k in METRIC_KEYS:
        v = metrics[k]
        print(f"  {k.upper():8s} {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")
    print(f"\nPer-subject MSE  best (#{best_i:3d})={mses[best_i]:.4f}  "
          f"median (#{median_i:3d})={mses[median_i]:.4f}  "
          f"worst (#{worst_i:3d})={mses[worst_i]:.4f}")

    # ── machine-readable metrics (subject means) for automated sweeps ──────────
    if args.metrics_out:
        import json
        summary = {k: float(np.nanmean(metrics[k])) for k in METRIC_KEYS}
        summary["n"] = int(len(ds))
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics JSON → {args.metrics_out}")

    if args.no_figures:
        return

    # ── figure: 3 subjects × 3 planes, each plane = [real | recon | error] ──
    subjects = [
        (best_i,   f"Best    MSE={mses[best_i]:.4f}"),
        (median_i, f"Median  MSE={mses[median_i]:.4f}"),
        (worst_i,  f"Worst   MSE={mses[worst_i]:.4f}"),
    ]
    planes = [
        ("Axial",     2),
        ("Coronal",   1),
        ("Sagittal",  0),
    ]

    n_subj, n_planes, n_cols = len(subjects), len(planes), 3   # real/recon/err
    fig, axes = plt.subplots(
        n_subj * n_planes, n_cols,
        figsize=(n_cols * 3.5, n_subj * n_planes * 3.0),
    )

    for row_base, (subj_idx, subj_label) in enumerate(subjects):
        real_vol  = reals[subj_idx]
        recon_vol = recons[subj_idx]
        err_vol   = np.abs(real_vol - recon_vol)

        for plane_offset, (plane_name, axis) in enumerate(planes):
            row = row_base * n_planes + plane_offset
            real_sl  = mid(real_vol,  axis)
            recon_sl = mid(recon_vol, axis)
            err_sl   = mid(err_vol,   axis)

            ax_real  = axes[row, 0]
            ax_recon = axes[row, 1]
            ax_err   = axes[row, 2]

            ax_real.imshow(real_sl,  cmap="hot", vmin=0, vmax=1, origin="lower")
            ax_recon.imshow(recon_sl, cmap="hot", vmin=0, vmax=1, origin="lower")
            im_err = ax_err.imshow(err_sl, cmap="Oranges", vmin=0, vmax=0.5, origin="lower")

            for ax in (ax_real, ax_recon, ax_err):
                ax.axis("off")

            if plane_offset == 0:
                ax_real.set_title(f"{subj_label}\nReal", fontsize=8)
                ax_recon.set_title(f"\nRecon", fontsize=8)
                ax_err.set_title(f"\n|Error|", fontsize=8)
                fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
            else:
                ax_real.set_title(plane_name, fontsize=7, color="gray")

    plt.suptitle(
        f"AE Reconstruction — Best / Median / Worst (MSE)  [{args.split} set, n={len(ds)}]",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    save_path = os.path.join(fig_dir, f"ae_recon_{args.split}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure → {save_path}")

    # ── per-subject MSE histogram ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(mses, bins=20, edgecolor="white", color="#2171b5")
    ax2.axvline(mses[best_i],   color="green",  linestyle="--", label=f"Best {mses[best_i]:.4f}")
    ax2.axvline(mses[median_i], color="orange", linestyle="--", label=f"Median {mses[median_i]:.4f}")
    ax2.axvline(mses[worst_i],  color="red",    linestyle="--", label=f"Worst {mses[worst_i]:.4f}")
    ax2.set_xlabel("Per-subject MSE (normalised [0,1] space)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"AE Reconstruction MSE Distribution [{args.split}, n={len(ds)}]")
    ax2.legend()
    plt.tight_layout()

    hist_path = os.path.join(fig_dir, f"ae_recon_mse_hist_{args.split}.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histogram → {hist_path}")


if __name__ == "__main__":
    main()
