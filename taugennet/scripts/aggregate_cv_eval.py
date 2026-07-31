#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
aggregate_cv_eval.py — aggregate per-fold evaluate_final metrics into a CV mean ± std.

Each evaluate_final run (per fold) writes a metrics.json via --metrics-out:
    {pearson, nrmse, ssim, mse, mae, n, fold}
This reads all fold_*/metrics.json under one or more experiment dirs and prints a
mean ± std (across folds) table — the proper CV result. Pass several experiment dirs
to compare them side by side (e.g. A/B off vs on, or conditioning a/b/c).

Usage:
  python scripts/aggregate_cv_eval.py results/records/eval/ab_emasnr_off \
                                       results/records/eval/ab_emasnr_on
  python scripts/aggregate_cv_eval.py results/records/eval/cond_a_mlp \
                                       results/records/eval/cond_b_map \
                                       results/records/eval/cond_c_ptau
"""
import argparse
import glob
import json
import os

METRICS = ["pearson", "nrmse", "ssim", "mse", "mae"]


def _mean_std(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(vals) / n
    std  = (sum((x - mean) ** 2 for x in vals) / n) ** 0.5 if n > 1 else 0.0
    return mean, std, n


def aggregate(exp_dir):
    """Return {metric: (mean, std, n_folds)} over fold_*/metrics.json in exp_dir."""
    paths = sorted(glob.glob(os.path.join(exp_dir, "fold_*", "metrics.json")))
    per_metric = {m: [] for m in METRICS}
    folds_found = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  warn: could not read {p}: {e}")
            continue
        folds_found.append(d.get("fold"))
        for m in METRICS:
            if d.get(m) is not None:
                per_metric[m].append(float(d[m]))
    return {m: _mean_std(v) for m, v in per_metric.items()}, folds_found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp_dirs", nargs="+",
                   help="One or more experiment record dirs (each holding fold_*/metrics.json).")
    args = p.parse_args()

    rows = []
    for d in args.exp_dirs:
        agg, folds = aggregate(d)
        name = os.path.basename(d.rstrip("/"))
        rows.append((name, agg, folds))

    name_w = max(len(n) for n, _, _ in rows) + 2
    header = f"{'experiment':<{name_w}}{'folds':>6}  " + "".join(f"{m.upper():>16}" for m in METRICS)
    print("\nCV results — mean ± std across folds (whole-brain, DK86-masked test)")
    print(header)
    print("-" * len(header))
    for name, agg, folds in rows:
        n_folds = max((agg[m][2] for m in METRICS), default=0)
        cells = ""
        for m in METRICS:
            mean, std, n = agg[m]
            cells += f"{mean:>7.4f}±{std:<7.4f}" if n else f"{'n/a':>16}"
        print(f"{name:<{name_w}}{n_folds:>6}  {cells}")
    print()


if __name__ == "__main__":
    main()
