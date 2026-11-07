#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Runs UNDER THE GITHUB taugennet root (TAUGENNET_ROOT must point at it) as a subprocess of
scripts/score_baseline.py. Two jobs:

1. Alignment guard: rebuild the held-out test set (or, with --split val, this fold's validation
   set) with the GitHub loader (src.dataset_final.build_dataloaders) and check it lists exactly
   the subjects the predictions were generated for, in the same order — the manifest's
   heldout_test subjects, or the generation's index.csv for val — so the subject_XXX.npy index
   means the same subject to both pipelines. Exits 2 on any mismatch so nothing gets scored
   against the wrong subjects.
2. per_subject_wholebrain.csv in the convention of the repo's analysis/make_per_subject.py:
   subject_idx, RID, group, pearson/mse/rmse/mae over the DK86 mask in SUVR space.

Standalone on purpose (no `training` imports): the local and GitHub taugennet trees both
provide a top-level package named `src`, so the GitHub one must be imported in its own process.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


def norm_rid(v):
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--gen-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--use-controls-322", action="store_true")
    p.add_argument("--controls-dir", default=None)
    p.add_argument("--split", choices=["test", "val"], default="test")
    a = p.parse_args()

    root = os.environ["TAUGENNET_ROOT"]
    sys.path.insert(0, root)
    from src.dataset_final import build_dataloaders  # noqa: E402  (the GitHub version)

    kw = dict(mode="atrophy", use_dk_mask=True)
    if a.fold is not None:
        kw.update(fold_idx=a.fold, n_folds=a.n_folds)
    if a.use_controls_322:
        kw.update(use_controls_322=True, controls_dir=a.controls_dir)
    _, va, te, *_ = build_dataloaders(**kw)
    if a.split == "val":
        te = va

    rid_of = lambda path: norm_rid(re.search(r"RID[_-](\d+)", path).group(1))  # noqa: E731
    gh_rids = [rid_of(pth) for pth in te.pet_paths]
    df = pd.read_csv(a.manifest)
    df["RID"] = df["RID"].map(norm_rid)
    if a.split == "test":
        man = df[df["split"] == "heldout_test"].reset_index(drop=True)
    else:                                       # the validation subjects, in generation order
        order = pd.read_csv(os.path.join(a.gen_dir, "index.csv"))["RID"].map(norm_rid)
        man = df.set_index("RID").loc[list(order)].reset_index()
    man_rids = list(man["RID"])
    if gh_rids != man_rids:
        only_gh, only_man = sorted(set(gh_rids) - set(man_rids)), sorted(set(man_rids) - set(gh_rids))
        print(f"[gh_align] MISMATCH: GitHub loader {a.split} set has {len(gh_rids)} subjects, reference {len(man_rids)}.\n"
              f"  only in GitHub loader: {only_gh[:20]}\n  only in manifest: {only_man[:20]}\n"
              f"  same membership but different order: {set(gh_rids) == set(man_rids)}", file=sys.stderr)
        sys.exit(2)
    n_npy = len([f for f in os.listdir(a.gen_dir) if re.fullmatch(r"subject_\d{3}\.npy", f)])
    if n_npy != len(te):
        print(f"[gh_align] MISMATCH: {n_npy} cached volumes in {a.gen_dir} but {len(te)} {a.split} subjects", file=sys.stderr)
        sys.exit(2)
    print(f"[gh_align] OK: GitHub loader {a.split} set == {'manifest heldout_test' if a.split == 'test' else 'generated index.csv'} "
          f"({len(te)} subjects, same order)")

    # per-subject whole-brain metrics, SUVR, DK86 mask (analysis/make_per_subject.py convention)
    mask = te._dk_mask.squeeze().numpy() > 0
    grp = dict(zip(man["RID"], man["group"]))
    rows = []
    for i in range(len(te)):
        pet, _, _ = te[i]                       # populates te._pet_norms[i] (masked min/max, SUVR)
        lo, hi = te._pet_norms[i]
        rng = float(hi) - float(lo)
        real = pet.squeeze().numpy()[mask] * rng + float(lo)
        gen = np.load(os.path.join(a.gen_dir, f"subject_{i:03d}.npy")).squeeze()[mask] * rng + float(lo)
        d = real - gen
        mse = float(np.mean(d ** 2))
        rows.append({"subject_idx": f"subject_{i:03d}", "RID": gh_rids[i], "group": grp.get(gh_rids[i], "?"),
                     "pearson_wholebrain": float(np.corrcoef(real, gen)[0, 1]), "mse_wholebrain": mse,
                     "rmse_wholebrain": float(np.sqrt(mse)), "mae_wholebrain": float(np.mean(np.abs(d)))})
    os.makedirs(a.out_dir, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out_dir, "per_subject_wholebrain.csv"), index=False)
    print(f"[gh_align] per-subject (SUVR): n={len(out)} r={out.pearson_wholebrain.mean():.4f}±{out.pearson_wholebrain.std():.4f} "
          f"mse={out.mse_wholebrain.mean():.4f} mae={out.mae_wholebrain.mean():.4f} → per_subject_wholebrain.csv")


if __name__ == "__main__":
    main()
