#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
The manuscript table for one model, in the summer report's layout: every evaluation level
(SET1 voxel, SET2 pooled regional, SET3 regional cross-subject, subject-level) × MSE / RMSE /
MAE / Pearson r, with a 5-fold CV column and a held-out test column — every cell with an SD
where one exists (Anil, 2026-10-29).

  5-fold CV      each fold scored on its own validation subjects (<fold>_val records, written by
                 score_baseline.py --split val); mean ± SD across the 5 folds. Subject-level
                 cells average each fold's per-subject values first.
  held-out test  the final (refit) model on the 120 test subjects; SET1–3 are single numbers,
                 subject-level cells are mean ± SD across the test subjects.
RMSE = sqrt(mean MSE) of a run, as in the report. Subject R (mentor's whole-brain-mean
correlation across subjects) is added as its own row.

    python scripts/cv_table.py --cohort cohort_v2 --cv-tag denseunet_paper --test-tag denseunet_paper_refit
Writes cv_table.{md,csv} in the test run's records dir.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.config import RECORDS_ROOT  # noqa: E402
from report_run import dk86_voxel_counts, regional_stats  # noqa: E402

LEVELS = [("Voxel-level (SET1)", "pooled_wholebrain_atrophy.csv"),
          ("Pooled regional (SET2)", "region_pooled_atrophy.csv"),
          ("Regional cross-subject (SET3)", "region_crosssubject_metrics_atrophy.csv")]
SUBJ = [("MSE", "mse_wholebrain"), ("RMSE", "rmse_wholebrain"), ("MAE", "mae_wholebrain"), ("Pearson r", "pearson_wholebrain")]


def run_metrics(d, w):
    """{(level, metric): value} for one scored run; subject-level also keeps the per-subject column."""
    out = {}
    for level, fn in LEVELS:
        t = pd.read_csv(os.path.join(d, fn), index_col=0)
        mse = float(t["mse"].mean())
        out[(level, "MSE")], out[(level, "RMSE")] = mse, float(np.sqrt(mse))
        out[(level, "MAE")], out[(level, "Pearson r")] = float(t["mae"].mean()), float(t["pearson"].mean())
    ps = pd.read_csv(os.path.join(d, "per_subject_wholebrain.csv"))
    for lab, col in SUBJ:
        out[("Subject-level", lab)] = ps[col].to_numpy(float)
    real = pd.read_csv(os.path.join(d, "regional_means_real_atrophy.csv"), index_col=0).to_numpy(float)
    gen = pd.read_csv(os.path.join(d, "regional_means_gen_atrophy.csv"), index_col=0).to_numpy(float)
    out[("Subject-level", "Subject R")] = regional_stats(real, gen, w)[0]
    return out, len(ps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--cv-tag", required=True, help="tag of the 5 CV folds (records <tag>/fold_k_val)")
    p.add_argument("--test-tag", required=True, help="tag of the final model (records <tag>/single)")
    p.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--records-root", default=RECORDS_ROOT)
    a = p.parse_args()
    base = os.path.join(a.records_root, a.cohort)
    w = dk86_voxel_counts()
    cv = []
    for f in a.folds:
        d = os.path.join(base, a.cv_tag, f"fold_{f}_val")
        if not os.path.exists(os.path.join(d, "per_subject_wholebrain.csv")):
            sys.exit(f"fold {f} validation predictions are not scored yet: {d}")
        cv.append(run_metrics(d, w))
    test_dir = os.path.join(base, a.test_tag, "single")
    test, n_test = run_metrics(test_dir, w)

    rows = []
    for key in test:
        fold_vals = [np.mean(m[key]) for m, _ in cv]        # per fold: the value, or its subjects' mean
        t = test[key]
        rows.append(dict(level=key[0], metric=key[1], cv_mean=np.mean(fold_vals), cv_sd=np.std(fold_vals, ddof=1),
                         test=np.mean(t), test_sd=np.std(t, ddof=1) if np.ndim(t) else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(test_dir, "cv_table.csv"), index=False, float_format="%.4f")
    n_val = "/".join(str(n) for _, n in cv)
    with open(os.path.join(test_dir, "cv_table.md"), "w") as fh:
        fh.write(f"| Evaluation level | Metric | 5-fold CV (mean ± SD; val n = {n_val}) | Held-out test (n = {n_test}) |\n"
                 "|---|---|---|---|\n")
        prev = None
        for r in df.itertuples():
            test_cell = f"{r.test:.4f}" + ("" if np.isnan(r.test_sd) else f" ± {r.test_sd:.4f}")
            fh.write(f"| {r.level if r.level != prev else ''} | {r.metric} | {r.cv_mean:.4f} ± {r.cv_sd:.4f} | {test_cell} |\n")
            prev = r.level
    print(open(os.path.join(test_dir, "cv_table.md")).read())
    print(f"→ {test_dir}/cv_table.{{md,csv}}")


if __name__ == "__main__":
    main()
