#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Headline numbers of one scored run, for reports and emails: the mentor's canonical Pearson r's
in SUVR (SET1 pooled voxels, SET3 regional cross-subject, Subject R = DK86 whole-brain mean SUVR
correlated across subjects) plus SET2 and the per-subject whole-brain r by diagnosis. They are
given for all test subjects and again for those with a measured plasma p-tau217 — the subset
TauGenNet sees as published. SET1/SET2 need voxels, so the subset gets only the regional ones.

    python scripts/report_run.py --cohort cohort_v2 --tag denseunet_paper
    python scripts/report_run.py --cohort cohort_v2 --tag denseunet --fold 0
Writes report.json next to the run's metrics.json.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.config import RECORDS_ROOT  # noqa: E402

GH = "/scratch/network/sz3962/tau-pet-synthesis-baselines/taugennet"
FLUID_CSV = "/scratch/network/sz3962/taugennet/data/raw/ADNI34Tau_withFluidBiomarkers.csv"
GROUPS = ["CN", "MCI", "AD"]


def dk86_voxel_counts():
    """Voxels per DK86 label in the evaluator's own label volume (weights of the whole-brain mean)."""
    os.environ["TAUGENNET_ROOT"] = GH
    sys.path[:0] = [GH, GH + "/scripts"]
    spec = importlib.util.spec_from_file_location("ef", GH + "/scripts/evaluate_final.py")
    ef = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ef)
    lab = ef._dk86_label_volume(None)
    return np.array([(lab == k).sum() for k in range(1, 87)], float)


def regional_stats(real, gen, w):
    """Subject R and SET3 from the evaluator's regional means (subjects × 86 regions)."""
    ok = np.isfinite(real).all(0) & np.isfinite(gen).all(0)
    mr = (real[:, ok] * w[ok]).sum(1) / w[ok].sum()
    mg = (gen[:, ok] * w[ok]).sum(1) / w[ok].sum()
    set3 = [np.corrcoef(real[:, j], gen[:, j])[0, 1] for j in np.flatnonzero(ok)]
    return float(np.corrcoef(mr, mg)[0, 1]), float(np.nanmean(set3))


def per_group(ps):
    out = {"all": (float(ps.pearson_wholebrain.mean()), len(ps))}
    for g in GROUPS:
        s = ps[ps.group == g]
        out[g] = (float(s.pearson_wholebrain.mean()), len(s))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--fold", type=int, default=None, help="CV fold (default: the single run)")
    p.add_argument("--records-root", default=RECORDS_ROOT)
    args = p.parse_args()
    d = os.path.join(args.records_root, args.cohort, args.tag,
                     "single" if args.fold is None else f"fold_{args.fold}")
    m = json.load(open(os.path.join(d, "metrics.json")))
    ps = pd.read_csv(os.path.join(d, "per_subject_wholebrain.csv"))
    real = pd.read_csv(os.path.join(d, "regional_means_real_atrophy.csv"), index_col=0)
    gen = pd.read_csv(os.path.join(d, "regional_means_gen_atrophy.csv"), index_col=0)
    assert list(real.index) == list(ps.subject_idx) == list(gen.index), "subject order differs"
    w = dk86_voxel_counts()

    fluid = pd.read_csv(FLUID_CSV, usecols=["RID", "pT217_F"], low_memory=False)
    plasma = set(fluid.dropna(subset=["pT217_F"]).RID.astype(int))
    has = ps.RID.astype(int).isin(plasma).to_numpy()

    rep = {"run": d, "n": int(m["n"]),
           "set1": m["set1_pooled_suvr"]["pearson"],
           "set2": m["set2_region_pooled_pearson_mean"],
           "set3": m["set3_region_crosssubject_pearson_mean"]}
    rep["subject_r"], set3_check = regional_stats(real.to_numpy(float), gen.to_numpy(float), w)
    assert abs(set3_check - rep["set3"]) < 1e-6, f"SET3 recomputed {set3_check} != evaluator {rep['set3']}"
    rep["per_subject_r"] = per_group(ps)
    rep["plasma_subset"] = {"n": int(has.sum())}
    rep["plasma_subset"]["subject_r"], rep["plasma_subset"]["set3"] = regional_stats(
        real.to_numpy(float)[has], gen.to_numpy(float)[has], w)
    rep["plasma_subset"]["per_subject_r"] = per_group(ps[has])
    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)

    print(f"{args.tag} ({os.path.basename(d)}), n = {rep['n']} test subjects")
    print(f"  SET1 pooled voxel r {rep['set1']:.3f} | SET2 {rep['set2']:.3f} | "
          f"SET3 regional r {rep['set3']:.3f} | Subject R {rep['subject_r']:.3f}")
    for name, s in [("all test", rep), (f"measured plasma (n = {rep['plasma_subset']['n']})", rep["plasma_subset"])]:
        g = s["per_subject_r"]
        print(f"  per-subject r, {name}: " + " | ".join(f"{k} {v[0]:.3f} (n={v[1]})" for k, v in g.items()))
    ss = rep["plasma_subset"]
    print(f"  measured-plasma subset: SET3 {ss['set3']:.3f} | Subject R {ss['subject_r']:.3f}")


if __name__ == "__main__":
    main()
