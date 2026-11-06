#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""localization_metrics.py — rank runs by WHERE tau goes, not just inter-subject burden.

evaluate_final.py already writes, per run:
    regional_means_real_<mode>.csv   (rows = subjects, cols = 86 DK regions)
    regional_means_gen_<mode>.csv

From those this computes three different questions:

  SET3  (per-region, across subjects)  – for each region, correlate real vs gen across subjects.
                                         Rewards getting each subject's overall/regional BURDEN
                                         LEVEL right. This is the mentor's mandated metric, but it
                                         is an INTER-subject measure.

  WITHIN (per-subject, across regions) – for each subject, correlate the 86 real regional means
                                         against the 86 generated ones. "Inside this brain, is tau
                                         distributed over the right regions?"

  PEAK   (per-subject, top-K regions)  – same as WITHIN but restricted to that subject's top-K
                                         highest-tau REAL regions. "Does the model rank the HOT
                                         regions correctly?" This is the sharpest localization test
                                         and the one with the most headroom.

Usage:
    localization_metrics.py results/records/cond4_spade/fold_0 [more dirs ...]
    localization_metrics.py --top-k 10 results/records/*/fold_0
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def _load_pair(rec_dir):
    real = glob.glob(os.path.join(rec_dir, "regional_means_real_*.csv"))
    gen  = glob.glob(os.path.join(rec_dir, "regional_means_gen_*.csv"))
    if not real or not gen:
        return None, None
    R = pd.read_csv(real[0], index_col=0)
    G = pd.read_csv(gen[0],  index_col=0)
    R, G = R.align(G, join="inner", axis=1)   # same regions
    R, G = R.align(G, join="inner", axis=0)   # same subjects
    return R.values.astype(float), G.values.astype(float)


def _safe_r(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return pearsonr(a, b)[0]


def metrics(Rv, Gv, top_k=10):
    n_subj, n_reg = Rv.shape
    set3   = np.nanmean([_safe_r(Rv[:, j], Gv[:, j]) for j in range(n_reg)])
    within = np.nanmean([_safe_r(Rv[i, :], Gv[i, :]) for i in range(n_subj)])
    peaks = []
    for i in range(n_subj):
        idx = np.argsort(Rv[i, :])[-top_k:]
        peaks.append(_safe_r(Rv[i, idx], Gv[i, idx]))
    return set3, within, np.nanmean(peaks), n_subj, n_reg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="records dirs containing regional_means_{real,gen}_*.csv")
    p.add_argument("--top-k", type=int, default=10,
                   help="how many of each subject's hottest REAL regions define the peak test")
    p.add_argument("--sort-by", choices=["peak", "within", "set3"], default="peak")
    a = p.parse_args()

    rows = []
    for d in a.dirs:
        Rv, Gv = _load_pair(d)
        if Rv is None:
            print(f"  [skip] no regional_means_*.csv in {d}", file=sys.stderr)
            continue
        s3, wi, pk, ns, nr = metrics(Rv, Gv, a.top_k)
        rows.append({"run": d, "SET3": s3, "WITHIN": wi, f"PEAK@{a.top_k}": pk, "n_subj": ns})
    if not rows:
        sys.exit("no runs with regional_means CSVs found")

    key = {"peak": f"PEAK@{a.top_k}", "within": "WITHIN", "set3": "SET3"}[a.sort_by]
    df = pd.DataFrame(rows).sort_values(key, ascending=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nranked by {key}.  SET3 = inter-subject (per region).  "
          f"WITHIN/PEAK = intra-subject localization (per subject).")


if __name__ == "__main__":
    main()
