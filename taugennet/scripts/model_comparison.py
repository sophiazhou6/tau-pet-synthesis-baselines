"""
model_comparison.py — V1 vs paper-architecture model comparison figure.

Produces: results/figures/model_comparison.png

Metrics shown (all per-subject mean ± std):
  MSE   — mean squared error per voxel
  SSIM  — structural similarity index
  NRMSE — |mean_real - mean_gen| / mean_real  (whole-image, per subject)
  PSNR  — 10*log10(1 / MSE_i) in dB  (per subject, then averaged)

NRMSE and PSNR require re-running evaluate.py (updated to compute them).
Placeholder (None) entries will be skipped in the figure.

After running updated eval_{all,paper}.slurm, fill in the Mean NRMSE/PSNR
values printed as "Mean NRMSE: X ± Y" and "Mean PSNR: X ± Y".
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Per-subject metrics (mean, std) ──────────────────────────────────────────
# Source: eval_all_3250697.out  (V1 models — Mean* = all test subjects after fix)
#         eval_paper_3252131.out (Paper models — Mean* = all test subjects)
#
# NRMSE / PSNR: fill in after re-running updated evaluate.py.
# Set to None to omit that panel for a given model.

MODELS = {
    "V1\nAtrophy\n(72spl)": {
        "n_test": 70,
        "mse":   (0.0055, 0.0029),
        "ssim":  (0.5743, 0.1033),
        "nrmse": (None, None),   # TODO: re-run eval
        "psnr":  (None, None),
    },
    "V1\np-tau217\n(72spl)": {
        "n_test": 52,
        "mse":   (0.0047, 0.0021),
        "ssim":  (0.6270, 0.0388),
        "nrmse": (None, None),
        "psnr":  (None, None),
    },
    "V1\nCombined\n(80spl)": {
        "n_test": 19,
        "mse":   (0.0061, 0.0023),
        "ssim":  (0.5679, 0.0412),
        "nrmse": (None, None),
        "psnr":  (None, None),
    },
    "Paper\nAtrophy": {
        "n_test": 70,
        "mse":   (0.0079, 0.0072),
        "ssim":  (0.3855, 0.1333),
        "nrmse": (None, None),
        "psnr":  (None, None),
    },
    "Paper\np-tau217": {
        "n_test": 52,
        "mse":   (0.0065, 0.0054),
        "ssim":  (0.4928, 0.0914),
        "nrmse": (None, None),
        "psnr":  (None, None),
    },
    "Paper\nCombined": {
        "n_test": 37,
        "mse":   (0.0117, 0.0105),
        "ssim":  (0.3163, 0.1110),
        "nrmse": (None, None),
        "psnr":  (None, None),
    },
}

# ── Colors: V1 = cool blues, Paper = warm reds ───────────────────────────────
COLORS = [
    "#4C72B0", "#5B8DB8", "#6AAEC7",   # V1: atrophy, ptau217, combined
    "#C44E52", "#DD8452", "#937860",   # Paper: atrophy, ptau217, combined
]

ARROW = "↓"   # ↓ lower is better
ARROW_UP = "↑"  # ↑ higher is better


def _panel(ax, metric_key, ylabel, title, lower_is_better=True):
    """Draw one grouped bar panel for a given metric key."""
    means, stds, labels, colors = [], [], [], []
    for (label, cfg), color in zip(MODELS.items(), COLORS):
        m, s = cfg[metric_key]
        if m is None:
            continue
        means.append(m)
        stds.append(s)
        labels.append(f"{label}\n(n={cfg['n_test']})")
        colors.append(color)

    if not means:
        ax.text(0.5, 0.5, "Pending re-run of evaluate.py",
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
                color="gray", style="italic")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        return

    x = np.arange(len(means))
    bars = ax.bar(x, means, color=colors, alpha=0.85, width=0.6)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=4, linewidth=1.2)

    arrow = ARROW if lower_is_better else ARROW_UP
    ax.set_title(f"{title}  {arrow}", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{ylabel}\n(per-subject mean ± std)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bar, v, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s * 1.05,
                f"{v:.4f}" if metric_key != "psnr" else f"{v:.1f}",
                ha="center", va="bottom", fontsize=7, rotation=0)

    # Dashed line separating V1 from Paper
    if len(means) == 6:
        ax.axvline(2.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(0.95, 0.97, "V1", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#4C72B0", alpha=0.7)
        ax.text(0.97, 0.97, "Paper", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#C44E52", alpha=0.7)


def make_figure(save_path="results/figures/model_comparison.png"):
    metrics = [
        ("mse",   "MSE",   "Mean Squared Error",            True),
        ("ssim",  "SSIM",  "Structural Similarity (SSIM)",  False),
        ("nrmse", "NRMSE", "Normalised RMSE",               True),
        ("psnr",  "PSNR",  "Peak SNR (dB)",                 False),
    ]

    # Only show panels where at least one model has a value
    active = [(k, yl, t, lib) for k, yl, t, lib in metrics
              if any(cfg[k][0] is not None for cfg in MODELS.values())]

    n_panels = max(len(active), 2)   # always at least 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (key, ylabel, title, lib) in zip(axes, active):
        _panel(ax, key, ylabel, title, lower_is_better=lib)

    fig.suptitle(
        "TauGenNet — V1 vs. Paper Architecture\n"
        "Per-subject mean ± std across test subjects",
        fontsize=12, y=1.02
    )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#4C72B0", label="V1 Atrophy (72split)"),
        Patch(color="#5B8DB8", label="V1 p-tau217 (72split)"),
        Patch(color="#6AAEC7", label="V1 Combined (80split)"),
        Patch(color="#C44E52", label="Paper Atrophy"),
        Patch(color="#DD8452", label="Paper p-tau217"),
        Patch(color="#937860", label="Paper Combined"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    make_figure()
