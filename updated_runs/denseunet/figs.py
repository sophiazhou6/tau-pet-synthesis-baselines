"""
Shared SUVR figure plotting for the baseline (used by scripts/suvr_sets.py and the
whole-brain scorer). All figures save with bbox_inches="tight" so titles, colorbars
and axis labels are never clipped (fixes the earlier cut-off scatter).
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _new(**kw):
    """Create a figure with matplotlib's DEFAULT (white) style, immune to any global
    style state a caller may have set (e.g. glass_brain_final's dark_background)."""
    plt.rcdefaults()
    return plt.subplots(**kw)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")   # tight bbox → nothing clipped
    plt.close(fig)


def plot_pooled_scatter(r_all, g_all, R1, out_path, tag=""):
    """SET 1 — density hexbin of pooled real vs predicted SUVR voxels."""
    vmax = float(np.percentile(r_all, 99.5)) if r_all.size else 1.0
    fig, ax = _new(figsize=(6.4, 6.4))
    hb = ax.hexbin(r_all, g_all, gridsize=80, bins="log", cmap="viridis",
                   extent=(0, vmax, 0, vmax))
    ax.plot([0, vmax], [0, vmax], "r--", lw=1.2, label="identity")
    ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
    ax.set_xlabel("Real tau SUVR"); ax.set_ylabel("Predicted tau SUVR")
    ax.set_title(f"SET 1 — Pooled voxel-level, whole region{(' ' + tag) if tag else ''}\n"
                 f"Pearson R = {R1:.3f}", pad=12)
    fig.colorbar(hb, ax=ax, label="log10(voxel count)")
    ax.legend(loc="upper left")
    _save(fig, out_path)


def plot_region_bars(pearson_by_region, title, out_path):
    """Horizontal bar of per-region Pearson R (sorted), for SET 2 or SET 3."""
    s = pearson_by_region.sort_values()
    colors = plt.cm.RdYlGn((s.values - (-0.3)) / (0.7 - (-0.3)))
    fig, ax = _new(figsize=(8, 16))
    ax.barh(range(len(s)), s.values, color=colors)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=5)
    ax.axvline(0, color="k", lw=0.6)
    ax.axvline(float(s.mean()), color="navy", ls="--", lw=1, label=f"mean R = {s.mean():.3f}")
    ax.set_xlabel("Pearson R (SUVR)"); ax.set_title(title, pad=10)
    ax.legend(loc="lower right"); ax.set_ylim(-1, len(s))
    _save(fig, out_path)


def plot_region_bars_ci(mean_series, std_series, title, out_path):
    """Per-region Pearson R as CV mean ± std (across folds), sorted horizontal bars."""
    order = mean_series.sort_values().index
    m = mean_series.loc[order]; s = std_series.loc[order]
    colors = plt.cm.RdYlGn((m.values - (-0.3)) / (0.7 - (-0.3)))
    fig, ax = _new(figsize=(8, 16))
    ax.barh(range(len(m)), m.values, xerr=s.values, color=colors,
            error_kw=dict(ecolor="0.35", elinewidth=0.6, capsize=1.5))
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index, fontsize=5)
    ax.axvline(0, color="k", lw=0.6)
    ax.axvline(float(m.mean()), color="navy", ls="--", lw=1, label=f"mean R = {m.mean():.3f}")
    ax.set_xlabel("Pearson R (SUVR), 5-fold mean ± std"); ax.set_title(title, pad=10)
    ax.legend(loc="lower right"); ax.set_ylim(-1, len(m))
    _save(fig, out_path)


def plot_pooled_vs_xsub(set2_pearson, set3_pearson, out_path):
    """Per-region SET 2 (pooled-voxel) vs SET 3 (cross-subject) R."""
    fig, ax = _new(figsize=(6.4, 6.4))
    ax.scatter(set2_pearson.values, set3_pearson.values, s=18, alpha=0.7)
    lim = [-0.3, 0.75]; ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("SET 2: pooled-voxel R (SUVR)")
    ax.set_ylabel("SET 3: cross-subject R (SUVR)")
    ax.set_title("Per-region: pooled-voxel vs cross-subject R (SUVR)", pad=10)
    _save(fig, out_path)


def plot_suvr_figures(r_all, g_all, set1_R, set2_df, set3_df, fig_dir, tag=""):
    """Emit all four SUVR figures into fig_dir. set2_df/set3_df indexed by region,
    with a 'pearson' column."""
    os.makedirs(fig_dir, exist_ok=True)
    plot_pooled_scatter(r_all, g_all, set1_R,
                        os.path.join(fig_dir, "pooled_wholebrain_scatter_SUVR.png"), tag=tag)
    plot_region_bars(set2_df["pearson"],
                     f"SET 2 — Pooled voxel-level Pearson R per region (SUVR){(' ' + tag) if tag else ''}",
                     os.path.join(fig_dir, "region_pooled_R_SUVR.png"))
    plot_region_bars(set3_df["pearson"],
                     f"SET 3 — Regional cross-subject Pearson R (SUVR){(' ' + tag) if tag else ''}",
                     os.path.join(fig_dir, "region_crosssubject_R_SUVR.png"))
    plot_pooled_vs_xsub(set2_df["pearson"], set3_df["pearson"],
                        os.path.join(fig_dir, "region_R_pooled_vs_xsub_SUVR.png"))
