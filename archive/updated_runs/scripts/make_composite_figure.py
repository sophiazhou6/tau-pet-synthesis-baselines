#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Report figure for one model run on the held-out test set, laid out like the summer FIGURE_*.png:

  A  9 test cases as glass brains, real over generated, one shared "Tau SUVR" colour bar —
     the sheet from tau-pet-synthesis-baselines/analysis/render_glass_9case_white.py (run it first)
  B  pooled voxel hexbin (SET1) — written by evaluate_final.py when the run is scored
  C  regional cross-subject correlation (SET3): Pearson r of each of the 86 DK regions — drawn here

    python scripts/make_composite_figure.py --run denseunet_paper
    python scripts/make_composite_figure.py --run denseunet_paper --pick-cases   # RIDs for panel A
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines.config import FIGURES_ROOT, RECORDS_ROOT  # noqa: E402
from plot_subject_figures import PRETTY  # noqa: E402

TAUGENNET_DATA = os.path.join(os.environ.get("TAUGENNET_ROOT", "/scratch/network/sz3962/taugennet"), "data", "raw_v2")
GROUPS = ["CN", "MCI", "AD"]


def pick_cases(per_subj, n=3):
    """Per diagnosis, the n subjects whose per-subject r is closest to the group median (representative,
    not best-case), among subjects with a measured plasma p-tau217 so TauGenNet can show the same people."""
    fl = pd.read_csv(os.path.join(TAUGENNET_DATA, "ADNI34Tau_withFluidBiomarkers.csv"), low_memory=False)
    measured = set(fl.loc[pd.to_numeric(fl["pT217_F"], errors="coerce").notna(), "RID"].astype(str))
    ps = per_subj[per_subj.RID.astype(str).isin(measured)].copy()
    rows = []
    for g in GROUPS:
        sub = ps[ps.group == g].copy()
        sub["dist"] = (sub.pearson_wholebrain - sub.pearson_wholebrain.median()).abs()
        rows.append(sub.nsmallest(n, "dist").sort_values("RID"))
    return pd.concat(rows)[["RID", "group", "pearson_wholebrain"]].reset_index(drop=True)


def region_label(col):
    """CTX_RH_SUPERIORPARIETAL → "Rh Superiorparietal", LEFT_THALAMUS_PROPER → "Left Thalamus Proper"
    (the summer chart's labels)."""
    for pre, hemi in (("CTX_LH_", "Lh "), ("CTX_RH_", "Rh ")):
        if col.startswith(pre):
            return hemi + col[len(pre):].capitalize()
    return " ".join(w.capitalize() for w in col.split("_"))


def cv_name(n_folds):
    return "5-fold CV" if n_folds == 5 else f"CV, {n_folds} of 5 folds"


def panel_region_r(r, title, path, cv=None, cv_subjects="validation subjects"):
    """Summer panel C: held-out-test cross-subject Pearson r of each DK region (SET3), one red diamond
    per region, highest at the top, dashed line at the test mean. With `cv` (regions × folds: each CV
    fold's r on its own validation subjects) every region also gets the folds' mean ± SD (ddof=1) as
    an error bar — the summer chart showed the same five values as a box."""
    r = r.dropna().sort_values()
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(8, 9.5))
    ax.hlines(y, -1, 1, color="#eeeeee", lw=0.6, zorder=0)
    ax.axvline(0, color="#999999", lw=0.8, zorder=1)
    ax.axvline(r.mean(), color="#c0392b", ls="--", lw=1.0, zorder=1, alpha=0.8)
    lo, hi = float(r.min()), float(r.max())
    if cv is not None:
        cv = cv.reindex(r.index)
        m, sd = cv.mean(axis=1), cv.std(axis=1, ddof=1)
        ax.axvline(m.mean(), color="#34495e", ls=":", lw=1.0, zorder=1)
        ax.errorbar(m.values, y, xerr=sd.values, fmt="o", ms=2.6, color="#34495e", ecolor="#7f8c8d",
                    elinewidth=0.8, capsize=1.4, zorder=2,
                    label=f"{cv_name(cv.shape[1])}, {cv_subjects}: mean ± SD (mean {m.mean():.3f})")
        lo, hi = min(lo, float((m - sd).min())), max(hi, float((m + sd).max()))
    ax.scatter(r.values, y, marker="D", s=22, color="#c0392b", zorder=3,
               label=f"Held-out TEST, final model (mean {r.mean():.3f})")
    ax.set_yticks(y); ax.set_yticklabels([region_label(c) for c in r.index], fontsize=4.5)
    ax.set_ylim(-1, len(r))
    ax.set_xlim(min(-0.2, lo - 0.05), max(0.8, hi + 0.05)); ax.set_xlabel("Pearson r", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.045), fontsize=7, frameon=False)   # below: no rows hidden
    ax.set_title(title, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help='"<tag>" (non-CV run, dir "single") or "<tag>:fold_<k>"')
    ap.add_argument("--cohort", default="cohort_v2")
    ap.add_argument("--panel-a", default=None,
                    help="9-case sheet (default: <figures>/<cohort>/glass_9case_whitebg/glass2row_<model>.png)")
    ap.add_argument("--pick-cases", action="store_true", help="print the 9 representative RIDs and exit")
    ap.add_argument("--out-root", default=None, help="default: results/figures/baselines/<cohort>/composite")
    ap.add_argument("--cv-tag", default=None,
                    help="tag of the 5 CV folds for panel C's error bars (records <tag>/fold_k_val); "
                         "default: the run's tag without _refit, if those records exist; 'none' to omit")
    ap.add_argument("--cv-exclude-folds", type=int, nargs="*", default=[],
                    help="folds left out of panel C's error bars (a failed generation, stated in the panel's title)")
    ap.add_argument("--cv-on", choices=["val", "test"], default="val",
                    help="subjects behind panel C's error bars: each fold's own validation subjects (default), or "
                         "test = the fold models on the held-out test subjects; test writes only a separate panel C file")
    a = ap.parse_args()
    tag, fn = a.run.split(":", 1) if ":" in a.run else (a.run, "single")
    name = PRETTY.get(tag, tag)
    rec = os.path.join(RECORDS_ROOT, a.cohort, tag, fn)
    if a.pick_cases:
        print(pick_cases(pd.read_csv(os.path.join(rec, "per_subject_wholebrain.csv"))).to_string(index=False))
        return
    out = os.path.join(a.out_root or os.path.join(FIGURES_ROOT, a.cohort, "composite"), tag)
    os.makedirs(out, exist_ok=True)
    panel_a = a.panel_a or os.path.join(FIGURES_ROOT, a.cohort, "glass_9case_whitebg",
                                        f"glass2row_{name.lower().replace('-', '').replace(' ', '_')}.png")
    panel_b = os.path.join(FIGURES_ROOT, a.cohort, tag, fn, "pooled_wholebrain_scatter_labeled.png")
    if not os.path.exists(panel_b):   # scripts/replot_pooled_scatter.py writes the one with the right inputs in its title
        panel_b = os.path.join(FIGURES_ROOT, a.cohort, tag, fn, "pooled_wholebrain_scatter_atrophy.png")
    for p in (panel_a, panel_b):
        if not os.path.exists(p):
            sys.exit(f"missing panel {p}")

    # C — SET3 per region
    import json
    n = json.load(open(os.path.join(rec, "metrics.json")))["n"]
    s3 = pd.read_csv(os.path.join(rec, "region_crosssubject_metrics_atrophy.csv"), index_col=0)["pearson"]
    cv_tag = a.cv_tag or (tag[:-len("_refit")] if tag.endswith("_refit") else None)
    cv = None
    if cv_tag and cv_tag != "none":
        folds = [k for k in range(5) if k not in a.cv_exclude_folds]
        sfx = "_val" if a.cv_on == "val" else ""
        files = {k: os.path.join(RECORDS_ROOT, a.cohort, cv_tag, f"fold_{k}{sfx}", "region_crosssubject_metrics_atrophy.csv")
                 for k in folds}
        if all(map(os.path.exists, files.values())):
            cv = pd.concat([pd.read_csv(f, index_col=0)["pearson"].rename(k) for k, f in files.items()], axis=1)
        else:
            print(f"[composite] no CV validation records for {cv_tag} — panel C without error bars")
    on_test = a.cv_on == "test"
    panel_c = os.path.join(out, "C_set3_region_r_folds_on_test.png" if on_test else "C_set3_region_r.png")
    bars = "" if cv is None else f"; error bars: {cv_name(cv.shape[1])} mean ± SD"
    if on_test:
        bars = "" if cv is None else "; error bars: the 5 fold models on these test subjects, mean ± SD"
    if cv is not None and a.cv_exclude_folds:
        bars += f" (fold {', '.join(map(str, a.cv_exclude_folds))} excluded: failed generation)"
    panel_region_r(s3, f"Regional cross-subject correlation (86 DK regions) — {name}\n"
                       f"final model, held-out test n={n}" + bars,
                   panel_c, cv, "same test subjects" if on_test else "validation subjects")
    if on_test:
        print(f"[composite] {name} panel C only → {panel_c}", flush=True)
        return

    imgs = [plt.imread(p) for p in (panel_a, panel_b, panel_c)]
    # A spans the top; B and C split the bottom row evenly by width,
    # the row as tall as the taller of the two, both panels top-aligned
    aA, aB, aC = (i.shape[0] / i.shape[1] for i in imgs)
    wB = 0.5
    W = 20.0
    hA, h2 = W * aA, max(W * wB * aB, W * (1 - wB) * aC)
    fig = plt.figure(figsize=(W, hA + h2))
    gs = fig.add_gridspec(2, 2, height_ratios=[hA, h2], width_ratios=[wB, 1 - wB], hspace=0.02, wspace=0.02)
    for (img, spec, letter) in [(imgs[0], gs[0, :], "A"), (imgs[1], gs[1, 0], "B"), (imgs[2], gs[1, 1], "C")]:
        ax = fig.add_subplot(spec)
        ax.imshow(img); ax.axis("off"); ax.set_anchor("N")
        ax.text(-0.01, 1.0, letter, transform=ax.transAxes, fontsize=26, fontweight="bold", va="top", ha="right")
    fig.savefig(os.path.join(out, f"FIGURE_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[composite] {name} → {out}/FIGURE_{tag}.png", flush=True)


if __name__ == "__main__":
    main()
