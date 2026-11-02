#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Roll per-fold metrics.json of every run under results/records/baselines/<cohort>/ into
one table: mean ± std across folds per model — the manuscript comparison table.

    python scripts/aggregate_baselines.py --cohort mentor235
    python scripts/aggregate_baselines.py --cohort cohort_v2 --tags denseunet swinunetr eagan
Writes summary.csv (long, one row per tag×metric) and summary.md (wide) next to the runs.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.config import RECORDS_ROOT  # noqa: E402

COLUMNS = [  # (label, extractor)
    ("SET1 R", lambda m: m["set1_pooled_suvr"]["pearson"]),
    ("SET2 R", lambda m: m["set2_region_pooled_pearson_mean"]),
    ("SET3 R", lambda m: m["set3_region_crosssubject_pearson_mean"]),
    ("NRMSE", lambda m: m["set1_pooled_suvr"]["nrmse"]),
    ("MSE", lambda m: m["set1_pooled_suvr"]["mse"]),
    ("MAE", lambda m: m["set1_pooled_suvr"]["mae"]),
    ("SSIM", lambda m: m["set1_pooled_suvr"]["ssim"]),
    ("WITHIN", lambda m: m["within_r_mean"]),
    ("PEAK@10", lambda m: m.get("peak10_r_mean", np.nan)),
    ("subj Pearson", lambda m: m["pearson"]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--tags", nargs="*", default=None)
    p.add_argument("--records-root", default=RECORDS_ROOT)
    args = p.parse_args()
    base = os.path.join(args.records_root, args.cohort)
    tags = args.tags or sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    rows, wide = [], []
    for tag in tags:
        files = sorted(f for f in glob.glob(os.path.join(base, tag, "fold_*", "metrics.json"))
                       if re.fullmatch(r"fold_\d+", os.path.basename(os.path.dirname(f))))
        files += glob.glob(os.path.join(base, tag, "single", "metrics.json"))
        if not files:
            continue
        ms = [json.load(open(f)) for f in files]
        n_test = ms[0]["n"]
        w = {"model": tag, "folds": len(ms), "n_test": n_test}
        for label, fn in COLUMNS:
            vals = np.array([fn(m) for m in ms], dtype=float)
            mean, std = float(np.nanmean(vals)), float(np.nanstd(vals)) if len(vals) > 1 else 0.0
            rows.append(dict(model=tag, metric=label, mean=mean, std=std, folds=len(vals), n_test=n_test))
            w[label] = f"{mean:.4f} ± {std:.4f}" if len(vals) > 1 else f"{mean:.4f}"
        wide.append(w)
    if not wide:
        sys.exit(f"no metrics.json found under {base}")
    long_df, wide_df = pd.DataFrame(rows), pd.DataFrame(wide)
    long_df.to_csv(os.path.join(base, "summary.csv"), index=False)
    try:
        table = wide_df.to_markdown(index=False)
    except ImportError:  # no `tabulate`
        cols = list(wide_df.columns)
        table = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n" + "\n".join(
            "| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in wide_df.iterrows())
    md = [f"# Baselines — cohort `{args.cohort}` (held-out test, DK86, SUVR; mean ± std over folds)\n", table]
    with open(os.path.join(base, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n→ {os.path.join(base, 'summary.md')}")


if __name__ == "__main__":
    main()
