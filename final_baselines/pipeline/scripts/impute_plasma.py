#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Plasma p-tau217 for TauGenNet's conditioning, with the missing values filled by multiple
imputation by chained equations (MICE; Sterne et al., BMJ 2009, doi 10.1136/bmj.b2393; White,
Royston & Wood, Stat Med 2011, doi 10.1002/sim.4067 — "the current gold standard", Erler et al.,
Stat Med 2016, doi 10.1002/sim.6944). Anil, 2026-10-29: "impute the missing plasma with a gold
standard method from the literature".

Per subject
  tau date     the earliest tau scan (SCANDATE_x) — the scan_01 PET the models predict
  p-tau217     pT217_F of the plasma draw closest to that date (EXAMDATE, else VISDATE); this also
               settles subjects whose draws disagree
  predictors   age at that scan, sex, education and MRI-derived FreeSurfer volumes of the AD-
               signature regions from the same scan row. Deliberately NOT used: diagnosis and
               cognitive scores (they would carry the labels into Anil's classifier), tau SUVR
               columns (the prediction target), amyloid PET / centiloids (a second PET scan).
               None of the 111 subjects without p-tau217 has any other plasma marker.

Per split (5 CV folds + the full-pool refit) the imputation model sees only that split's
training subjects; it is then applied to every subject (validation and test ones included),
so no held-out plasma value or label shapes another subject's value. 20 imputations
(IterativeImputer with BayesianRidge and posterior sampling, on z-scored features and log
p-tau217) are averaged into one value per subject, since TauGenNet is trained once per split.
Measured values are never changed. A masking check on the training subjects reports how well
the predictors recover held-back measured values.

    python scripts/impute_plasma.py --cohort cohort_v2
Writes <TAUGENNET_ROOT>/data/raw_v2/plasma_imputed/{fold_0..fold_4,refit}.csv + summary.json.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from denseunet.cohort import fold_indices, read_manifest  # noqa: E402
from train_baseline import resolve_manifest  # noqa: E402

DATA = os.path.join(os.environ.get("TAUGENNET_ROOT", "/scratch/network/sz3962/taugennet"), "data", "raw_v2")
VOLUMES = ["META_TEMPORAL_VOLUME", "CTX_ENTORHINAL_VOLUME", "CTX_INFERIORTEMPORAL_VOLUME",
           "CTX_MIDDLETEMPORAL_VOLUME", "CTX_FUSIFORM_VOLUME", "CTX_PARAHIPPOCAMPAL_VOLUME",
           "HIPPOCAMPUS_VOLUME", "AMYGDALA_VOLUME", "CTX_PRECUNEUS_VOLUME", "CTX_INFERIORPARIETAL_VOLUME",
           "LATERAL_VENTRICLE_VOLUME"]
FEATURES = ["AGE_AT_SCAN", "PTGENDER", "PTEDUCAT"] + VOLUMES
N_IMPUTATIONS = 20


def subject_table(fluid_csv, rids):
    d = pd.read_csv(fluid_csv, low_memory=False)
    d = d[d.RID.isin(rids)].copy()
    d["tau_date"] = pd.to_datetime(d.SCANDATE_x)
    d["draw_date"] = pd.to_datetime(d.EXAMDATE).fillna(pd.to_datetime(d.VISDATE))
    rows, conflicts = [], []
    for rid, g in d.groupby("RID"):
        first = g.tau_date.min()
        base = g[g.tau_date == first].iloc[0]
        rec = {"RID": int(rid), **{c: float(base[c]) for c in FEATURES}}
        p = g[g.pT217_F.notna()]
        if len(p):
            gap = (p.draw_date - first).abs().dt.days
            pick = p.loc[gap.idxmin()]
            rec.update(pT217_measured=float(pick.pT217_F), draw_gap_days=int(gap.min()))
            if p.pT217_F.nunique() > 1:
                conflicts.append({"RID": int(rid), "values": sorted(map(float, p.pT217_F.unique())),
                                  "chosen": float(pick.pT217_F)})
        else:
            rec.update(pT217_measured=np.nan, draw_gap_days=np.nan)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("RID"), conflicts


def impute(tab, train_rids, seeds=range(N_IMPUTATIONS)):
    """Mean of N MICE draws of p-tau217 for every subject, the model fit on train_rids only."""
    X = tab[FEATURES].copy()
    X["log_ptau"] = np.log(tab["pT217_measured"])
    mu, sd = X.loc[train_rids].mean(), X.loc[train_rids].std().replace(0, 1)
    Z = (X - mu) / sd
    draws = []
    for s in seeds:
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=25,
                               random_state=s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp.fit(Z.loc[train_rids].to_numpy())
            draws.append(imp.transform(Z.to_numpy())[:, -1])
    log_ptau = np.mean(draws, axis=0) * sd["log_ptau"] + mu["log_ptau"]
    out = pd.Series(np.exp(log_ptau), index=tab.index)
    measured = tab["pT217_measured"].notna()
    out[measured] = tab.loc[measured, "pT217_measured"]       # observed values untouched
    return out


def masking_check(tab, train_rids, frac=0.2, seed=0):
    """Hold back 20 % of the measured training values, impute them, compare (log scale)."""
    rng = np.random.default_rng(seed)
    meas = [r for r in train_rids if np.isfinite(tab.at[r, "pT217_measured"])]
    held = set(rng.choice(meas, size=int(frac * len(meas)), replace=False).tolist())
    t = tab.copy()
    t.loc[list(held), "pT217_measured"] = np.nan
    pred = impute(t, train_rids, seeds=range(5))
    true = np.log(tab.loc[list(held), "pT217_measured"]); est = np.log(pred[list(held)])
    return {"n_held": len(held), "r_log": float(np.corrcoef(true, est)[0, 1]),
            "r2_log": float(1 - ((true - est) ** 2).sum() / ((true - true.mean()) ** 2).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="cohort_v2")
    p.add_argument("--out-dir", default=os.path.join(DATA, "plasma_imputed"))
    a = p.parse_args()
    man = read_manifest(resolve_manifest(a.cohort), check_files=False)   # RID as str (fold_indices)
    rid = man["RID"].astype(int)
    tab, conflicts = subject_table(os.path.join(DATA, "ADNI34Tau_withFluidBiomarkers.csv"), set(rid))
    assert len(tab) == len(man), f"{len(man) - len(tab)} cohort subjects have no row in the plasma table"
    os.makedirs(a.out_dir, exist_ok=True)
    summary = {"features": FEATURES, "n_imputations": N_IMPUTATIONS, "conflicts": conflicts,
               "n_missing": int(tab.pT217_measured.isna().sum()), "splits": {}}
    splits = {f"fold_{k}": k for k in range(5)}
    splits["refit"] = None
    for name, k in splits.items():
        if k is None:
            tr = [i for i, s in enumerate(man["split"]) if s != "heldout_test"]
            va, te = [], [i for i, s in enumerate(man["split"]) if s == "heldout_test"]
        else:
            tr, va, te = fold_indices(man, fold_idx=k, verbose=False)
        role = pd.Series("train", index=rid.values)
        role.iloc[va] = "val"; role.iloc[te] = "test"
        train_rids = list(rid.iloc[tr])
        val_ptau = impute(tab, train_rids)
        out = pd.DataFrame({"RID": tab.index, "role": role.loc[tab.index].values,
                            "pT217_measured": tab.pT217_measured.values, "pT217": val_ptau.values,
                            "imputed": tab.pT217_measured.isna().values})
        out.to_csv(os.path.join(a.out_dir, f"{name}.csv"), index=False, float_format="%.5f")
        chk = masking_check(tab, train_rids)
        summary["splits"][name] = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
                                   "imputed_by_role": out[out.imputed].role.value_counts().to_dict(),
                                   "masking_check": chk}
        print(f"[impute] {name}: train {len(tr)} / val {len(va)} / test {len(te)}; imputed "
              f"{out[out.imputed].role.value_counts().to_dict()}; held-back check r(log) {chk['r_log']:.3f} "
              f"(R² {chk['r2_log']:.3f}, n={chk['n_held']})", flush=True)
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    gaps = tab.draw_gap_days.dropna()
    print(f"[impute] {len(conflicts)} subjects with disagreeing draws (closest to the tau scan kept); "
          f"draw-to-scan gap median {gaps.median():.0f} d, 90th pct {gaps.quantile(0.9):.0f} d → {a.out_dir}")


if __name__ == "__main__":
    main()
