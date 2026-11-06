#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
recreate_paper_figures.py — Recreate TauGenNet paper figures from cached SiLU results.

Paper: "Plasma-Driven Tau PET Image Synthesis via Text-Guided 3D Diffusion Models"
DOI: 10.1109/TRPMS.2026.3688162

Figures reproduced here (no GPU required — loads from pkl caches):
    Fig. 3  — Training loss curves  (val curve not saved in ckpt; train only)
    Table II — Ablation NRMSE & SSIM: MRI+cond vs. cond-only per ROI
    Table IV — NRMSE & SSIM across plasma p-tau217 ranges × brain region
    Fig. 5  — Boxplots: real vs. generated ROI values at different p-tau217 ranges

Not reproducible without GPU / additional models:
    Fig. 4  — Plasma sweep requires live inference (synthesize_tau_pet)
    Fig. 6  — Brain surface plots require FreeSurfer / nilearn surface projection
    Table III — Pix2pix comparison requires a trained pix2pix baseline

All outputs saved to: results/figures/paper_recreation/
"""

import os, sys, pickle
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

OUT_DIR = "results/figures/paper_recreation"
os.makedirs(OUT_DIR, exist_ok=True)

ROI_NAMES  = ["Parahippocampal", "Fusiform", "Inferior Temporal",
              "Hippocampus", "Post. Cingulate", "Entorhinal"]
ROI_SHORT  = ["Parahip.", "Fusiform", "Inf.Temp.", "Hippoc.", "Post.Cing.", "Entorh."]
PLASMA_BINS   = [(0, 2), (2, 4), (4, 6), (6, 8), (10, float("inf"))]
BIN_LABELS    = ["0–2", "2–4", "4–6", "6–8", "10+"]
PLASMA_SWEEP  = [0.65, 3.65, 6.65, 10.65]   # paper Fig. 4 values

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pkl(mode):
    path = f"results/records/{mode}_paper.pkl"
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found — skipping {mode}")
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def nrmse(real_vals, gen_vals):
    """NRMSE = |mean(real) − mean(gen)| / (mean(real) + ε)"""
    mr, mg = np.mean(real_vals), np.mean(gen_vals)
    return abs(mr - mg) / (mr + 1e-8)


def bin_label(ptau):
    for (lo, hi), lbl in zip(PLASMA_BINS, BIN_LABELS):
        if lo <= ptau < hi:
            return lbl
    return None


def roi_group(records, prefix, roi):
    """Collect per-subject ROI values for prefix in {real, full, only}."""
    return [r[f"{prefix}_{roi}"] for r in records if f"{prefix}_{roi}" in r]


# ── Fig. 3: Training loss curves ──────────────────────────────────────────────

import glob, re

def _parse_val_losses_from_log(mode):
    """Parse (epoch, val_loss) pairs from the most recent silu SLURM log."""
    pattern = f"results/logs/silu_{mode}_*.out"
    logs = sorted(glob.glob(pattern))
    if not logs:
        return [], None, None
    log = logs[-1]
    epochs, vals = [], []
    best_val, stop_epoch = None, None
    with open(log) as f:
        for line in f:
            m = re.search(r"Epoch\s+(\d+)/\d+.*?val=([\d.]+)", line)
            if m:
                epochs.append(int(m.group(1)))
                vals.append(float(m.group(2)))
            m2 = re.search(r"Early stopping at epoch (\d+).*Best val=([\d.]+)", line)
            if m2:
                stop_epoch = int(m2.group(1))
                best_val   = float(m2.group(2))
    return list(zip(epochs, vals)), best_val, stop_epoch


def fig3_training_curves():
    """Paper Fig. 3 — training and validation MSE curves (SiLU, early-stopping shown)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    modes  = ["atrophy", "ptau217", "combined"]
    titles = ["Atrophy conditioning", "Plasma p-tau217", "Combined conditioning"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, mode, title, color in zip(axes, modes, titles, colors):
        ckpt_path = f"results/checkpoints/silu/{mode}/diff_{mode}.pt"
        if not os.path.exists(ckpt_path):
            ax.text(0.5, 0.5, f"Checkpoint\nnot found\n({mode})",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(title, fontsize=10)
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        train_losses = ckpt.get("diff_losses", [])
        final_epoch  = ckpt.get("epoch", len(train_losses))

        val_pairs, best_val, stop_epoch = _parse_val_losses_from_log(mode)

        # Train curve (every epoch, from checkpoint)
        ax.plot(range(1, len(train_losses) + 1), train_losses,
                color=color, linewidth=1.2, alpha=0.85, label="Train MSE")

        # Val curve (every 10 epochs, from log)
        if val_pairs:
            v_epochs, v_vals = zip(*val_pairs)
            ax.plot(v_epochs, v_vals, color=color, linewidth=1.4,
                    linestyle="--", alpha=0.9, label="Val MSE")
            best_ep = v_epochs[int(np.argmin(v_vals))]
            ax.axvline(best_ep, color="red", linestyle=":", linewidth=1.2,
                       label=f"Best val (ep {best_ep})")

        # Early-stopping marker
        if stop_epoch:
            ax.axvline(stop_epoch, color="black", linestyle="-.", linewidth=1.0,
                       alpha=0.6, label=f"Early stop (ep {stop_epoch})")
            ax.text(stop_epoch, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.15,
                    f" ep {stop_epoch}", fontsize=7, color="black", va="top")

        subtitle = f"best val={best_val:.4f}" if best_val else f"epoch {final_epoch}"
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("MSE loss", fontsize=9)
        ax.set_title(f"{title}\n({subtitle})", fontsize=9)
        ax.legend(fontsize=7.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.suptitle("Fig. 3 — Training & validation MSE curves — SiLU model", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_training_curves.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved Fig. 3 → {path}")


# ── Table II: Ablation NRMSE & SSIM ──────────────────────────────────────────

def table2_ablation():
    """
    Paper Table II — NRMSE and 3D SSIM for ablation (MRI+cond vs. cond-only).
    Rendered as a grouped bar chart (NRMSE) + printed table.
    Note: 3D SSIM here is the global per-subject mean SSIM (paper uses 3D SSIM).
    """
    modes       = ["atrophy", "ptau217", "combined"]
    mode_labels = ["MRI + Atrophy / Atrophy-only",
                   "MRI + Plasma / Plasma-only",
                   "MRI + Both / Both-only"]
    full_colors = ["#4C72B0", "#DD8452", "#55A868"]
    only_colors = ["#A8C4E8", "#F5C49A", "#A8D4B4"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_nrmse, ax_ssim = axes

    x  = np.arange(len(ROI_SHORT))
    w  = 0.13
    offsets = np.linspace(-2.5 * w, 2.5 * w, 6)

    print("\n=== Table II — Ablation NRMSE per ROI ===")
    header = f"{'Mode':<20}" + "".join(f"{r:>12}" for r in ROI_SHORT)
    print(header)
    print("-" * len(header))

    all_bars = []
    for i, (mode, mlbl, fc, oc) in enumerate(
            zip(modes, mode_labels, full_colors, only_colors)):
        recs = load_pkl(mode)
        if not recs:
            continue
        nrmse_full, nrmse_only = [], []
        ssim_full_all = [r["ssim"] for r in recs]
        for roi in ROI_NAMES:
            rv = roi_group(recs, "real", roi)
            fv = roi_group(recs, "full", roi)
            ov = roi_group(recs, "only", roi)
            nrmse_full.append(nrmse(rv, fv))
            nrmse_only.append(nrmse(rv, ov))

        # Print table rows
        full_label = mlbl.split(" / ")[0]
        only_label = mlbl.split(" / ")[1]
        print(f"  {full_label:<18}" + "".join(f"{v:>12.4f}" for v in nrmse_full))
        print(f"  {only_label:<18}" + "".join(f"{v:>12.4f}" for v in nrmse_only))

        b1 = ax_nrmse.bar(x + offsets[2*i],   nrmse_full, w, label=f"{full_label}",
                          color=fc, alpha=0.88)
        b2 = ax_nrmse.bar(x + offsets[2*i+1], nrmse_only, w, label=f"{only_label}",
                          color=oc, alpha=0.88)
        all_bars += [b1, b2]

        # SSIM bars (overall, not per-ROI — paper uses 3D SSIM)
        ssim_mean = np.mean(ssim_full_all)
        ssim_std  = np.std(ssim_full_all)
        ax_ssim.bar(i, ssim_mean, color=fc, alpha=0.88, width=0.5,
                    yerr=ssim_std, capsize=5,
                    error_kw=dict(linewidth=1.2))
        ax_ssim.text(i, ssim_mean + ssim_std + 0.005,
                     f"{ssim_mean:.3f}±{ssim_std:.3f}",
                     ha="center", va="bottom", fontsize=8)

    ax_nrmse.set_xticks(x)
    ax_nrmse.set_xticklabels(ROI_SHORT, fontsize=9)
    ax_nrmse.set_ylabel("NRMSE", fontsize=10)
    ax_nrmse.set_title("Table II (left) — Ablation NRMSE per brain region", fontsize=10)
    ax_nrmse.legend(fontsize=7.5, ncol=2)

    ax_ssim.set_xticks(range(len(modes)))
    ax_ssim.set_xticklabels(["Atrophy\n+MRI", "Plasma\n+MRI", "Both\n+MRI"], fontsize=9)
    ax_ssim.set_ylabel("Mean SSIM (per-subject)", fontsize=10)
    ax_ssim.set_title("Table II (right) — SSIM by conditioning mode\n"
                      "(global 3D SSIM; paper uses per-ROI)", fontsize=10)

    fig.suptitle("Table II — Ablation: MRI+conditioning vs. conditioning-only — SiLU", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "table2_ablation.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved Table II → {path}")


# ── Table IV: NRMSE & SSIM across plasma ranges × ROI ────────────────────────

def table4_plasma_roi():
    """
    Paper Table IV — NRMSE and 3D SSIM across plasma p-tau217 ranges × brain region.
    Rendered as two heatmaps (NRMSE | SSIM) side by side.
    Data source: ptau217_paper.pkl (primary), combined_paper.pkl (secondary).
    """
    for mode in ["ptau217", "combined"]:
        recs = load_pkl(mode)
        if not recs or "ptau" not in recs[0]:
            continue

        nrmse_mat = np.full((len(BIN_LABELS), len(ROI_NAMES)), np.nan)
        ssim_mat  = np.full((len(BIN_LABELS), len(ROI_NAMES)), np.nan)
        bin_n     = {lbl: 0 for lbl in BIN_LABELS}

        for lbl_i, lbl in enumerate(BIN_LABELS):
            bin_recs = [r for r in recs if bin_label(r["ptau"]) == lbl]
            bin_n[lbl] = len(bin_recs)
            if not bin_recs:
                continue
            ssim_mean = np.mean([r["ssim"] for r in bin_recs])
            for roi_j, roi in enumerate(ROI_NAMES):
                rv = [r[f"real_{roi}"] for r in bin_recs]
                gv = [r[f"full_{roi}"] for r in bin_recs]
                nrmse_mat[lbl_i, roi_j] = nrmse(rv, gv)
                ssim_mat[lbl_i, roi_j]  = ssim_mean   # global SSIM, same for all ROIs in bin

        print(f"\n=== Table IV — {mode}: Subjects per plasma bin ===")
        for lbl in BIN_LABELS:
            print(f"  {lbl} pg/mL : {bin_n[lbl]}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        for ax, mat, cmap, title in [
            (ax1, nrmse_mat, "YlOrRd", "NRMSE"),
            (ax2, ssim_mat,  "YlGn",   "SSIM (global, per bin)"),
        ]:
            display = np.where(np.isnan(mat), 0, mat)
            vmax    = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1.0
            im      = ax.imshow(display, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
            for i, lbl in enumerate(BIN_LABELS):
                for j in range(len(ROI_NAMES)):
                    if np.isnan(mat[i, j]):
                        ax.text(j, i, "—", ha="center", va="center",
                                fontsize=10, color="gray")
                    else:
                        ax.text(j, i, f"{mat[i, j]:.3f}",
                                ha="center", va="center", fontsize=8.5)
                # n= annotation on left
                if bin_n[BIN_LABELS[i]] > 0:
                    ax.text(-0.7, i, f"n={bin_n[BIN_LABELS[i]]}",
                            ha="right", va="center", fontsize=7.5, color="dimgray")
            ax.set_xticks(range(len(ROI_SHORT)))
            ax.set_xticklabels(ROI_SHORT, fontsize=9, rotation=20, ha="right")
            ax.set_yticks(range(len(BIN_LABELS)))
            ax.set_yticklabels([f"{l} pg/mL" for l in BIN_LABELS], fontsize=9)
            ax.set_xlabel("Brain region")
            ax.set_ylabel("Plasma p-tau217 (pg/mL)")
            fig.colorbar(im, ax=ax, label=title, shrink=0.85)
            ax.set_title(title, fontsize=10)

        fig.suptitle(f"Table IV — NRMSE & SSIM across plasma ranges × ROI  ({mode} — SiLU)",
                     fontsize=11)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"table4_plasma_roi_{mode}.pdf")
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"Saved Table IV ({mode}) → {path}")


# ── Fig. 5: Boxplots real vs. generated by plasma bin ────────────────────────

def fig5_boxplots():
    """
    Paper Fig. 5 — Box plots of average ROI values for real and generated tau PET
    at different p-tau217 plasma ranges. One subplot per ROI (2×3 grid).
    Data: ptau217_paper.pkl and combined_paper.pkl.
    """
    for mode in ["ptau217", "combined"]:
        recs = load_pkl(mode)
        if not recs or "ptau" not in recs[0]:
            continue

        # Group per-subject ROI values by plasma bin
        real_bins = {lbl: {r: [] for r in ROI_NAMES} for lbl in BIN_LABELS}
        gen_bins  = {lbl: {r: [] for r in ROI_NAMES} for lbl in BIN_LABELS}
        for rec in recs:
            lbl = bin_label(rec["ptau"])
            if lbl is None:
                continue
            for roi in ROI_NAMES:
                real_bins[lbl][roi].append(rec[f"real_{roi}"])
                gen_bins[lbl][roi].append(rec[f"full_{roi}"])

        real_color = "#E8A0A0"
        gen_color  = "#A0C4E8"

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for ax, roi in zip(axes, ROI_NAMES):
            box_data   = []
            box_colors = []
            tick_lbls  = []
            for lbl in BIN_LABELS:
                r_vals = real_bins[lbl][roi]
                g_vals = gen_bins[lbl][roi]
                if r_vals or g_vals:
                    box_data.append(r_vals if r_vals else [0])
                    box_colors.append(real_color)
                    tick_lbls.append(f"Real\n{lbl}")
                    box_data.append(g_vals if g_vals else [0])
                    box_colors.append(gen_color)
                    tick_lbls.append(f"Gen\n{lbl}")

            if not box_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_title(roi, fontsize=10, fontweight="bold")
                continue

            bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                            medianprops=dict(color="black", linewidth=1.2))
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)

            # Mean marker as red dashed line
            for i, data in enumerate(box_data):
                if data:
                    ax.hlines(np.mean(data), i + 0.7, i + 1.3,
                              colors="red", linestyles="dashed", linewidth=1.0)

            ax.set_xticks(range(1, len(tick_lbls) + 1))
            ax.set_xticklabels(tick_lbls, fontsize=6.5, rotation=0)
            ax.set_ylabel("Mean SUVR in region", fontsize=8)
            ax.set_title(roi, fontsize=10, fontweight="bold")

        legend_els = [Patch(facecolor=real_color, label="Real tau PET"),
                      Patch(facecolor=gen_color,  label="Generated tau PET")]
        fig.legend(handles=legend_els, loc="upper right", fontsize=9)
        fig.suptitle(
            f"Fig. 5 — ROI average values: real vs. generated at different p-tau217 ranges\n"
            f"({mode} conditioning — SiLU model)",
            fontsize=11, y=1.01
        )
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"fig5_boxplots_{mode}.pdf")
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"Saved Fig. 5 ({mode}) → {path}")


# ── Summary metrics table ─────────────────────────────────────────────────────

def print_metrics_summary():
    print("\n" + "=" * 70)
    print("  SiLU Model — Test Set Metrics Summary (from cached pkl)")
    print("=" * 70)
    fmt_hdr = f"{'Mode':<12}  {'SSIM':>10}  {'MSE':>10}  {'PSNR (dB)':>10}  {'MAE':>10}  {'N':>4}"
    print(fmt_hdr)
    print("-" * len(fmt_hdr))
    for mode in ["atrophy", "ptau217", "combined"]:
        recs = load_pkl(mode)
        if not recs:
            continue
        ssims = [r["ssim"] for r in recs]
        mses  = [r["mse"]  for r in recs]
        psnrs = [r["psnr"] for r in recs]
        maes  = [r["mae"]  for r in recs]
        print(f"  {mode:<10}  "
              f"{np.mean(ssims):>6.4f}±{np.std(ssims):.4f}  "
              f"{np.mean(mses):>6.5f}±{np.std(mses):.5f}  "
              f"{np.mean(psnrs):>7.2f}±{np.std(psnrs):.2f}  "
              f"{np.mean(maes):>6.4f}±{np.std(maes):.4f}  "
              f"{len(recs):>4}")
    print("=" * 70)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}")
    print_metrics_summary()
    fig3_training_curves()
    table2_ablation()
    table4_plasma_roi()
    fig5_boxplots()
    print("\nDone. Figures not reproduced (require GPU/FreeSurfer):")
    print("  Fig. 4  — plasma sweep  (needs live inference)")
    print("  Fig. 6  — brain surface plots  (needs FreeSurfer / nilearn surface)")
    print("  Table III — pix2pix comparison  (needs trained pix2pix baseline)")
