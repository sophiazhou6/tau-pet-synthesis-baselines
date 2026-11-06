"""Phase 6 dataset — manifest-driven pooled cohort (AD + MCI + CN).

Differs from dataset_spatial only in HOW the cohort and split are chosen:
  * subjects/paths/split come from results/records/phase6_manifest_v3.csv
    (no directory scanning, no mentor split, one scan per subject already resolved)
  * PET is Eroded_CerebGry for ALL groups -> uniform SUVR reference
  * diagnosis is 3-class: CN=0, MCI=1, AD=2
  * cond_mode="combined" -> (87,) = [ptau217, 86 atrophy z-scores]

All tensor logic is reused from SpatialTauPETDataset. "combined" is delivered by
passing the 87-vector through the ptau slot (the class does not assume its shape).
"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .config import BASE_DIR, BATCH_SIZE, ADNI_FLUID_CSV, SEED
from .atlas_labels import load_atlas
from .dataset_spatial import (SpatialTauPETDataset, _build_rid_to_atrophy,
                              _build_rid_to_ptau, get_default_atlas_path)

DIAG = {"CN": 0, "MCI": 1, "AD": 2}
DEFAULT_MANIFEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "results", "records", "phase6_manifest_v3.csv")


def build_dataloaders(manifest=None, batch_size=BATCH_SIZE, seed=SEED, use_dk_mask=True,
                      atlas_path=None, fold_idx=0, n_folds=5, cond_mode="combined",
                      use_tissue=False, num_workers=2, no_val_split=False):
    manifest = manifest or DEFAULT_MANIFEST
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"Phase 6 manifest not found: {manifest}")
    atlas_path = atlas_path or get_default_atlas_path()
    atlas_data, _, inv_atlas_affine = load_atlas(atlas_path)
    print(f"Atlas {atlas_data.shape}, {(np.unique(atlas_data) > 0).sum()} labels", flush=True)

    df = pd.read_csv(manifest, dtype={"RID": str})
    df = df[df["fold"] == fold_idx].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for fold {fold_idx} in {manifest}")

    rid_to_atrophy = _build_rid_to_atrophy(BASE_DIR)
    rid_to_ptau = _build_rid_to_ptau(ADNI_FLUID_CSV)

    pet, mri, atr, diag, cond, split, grp, dropped = [], [], [], [], [], [], [], 0
    for _, r in df.iterrows():
        rid = str(r["RID"])
        if rid not in rid_to_atrophy or rid not in rid_to_ptau:
            dropped += 1
            continue
        if not (os.path.exists(r["pet"]) and os.path.exists(r["mri"])):
            dropped += 1
            continue
        a = np.asarray(rid_to_atrophy[rid], dtype=np.float32)          # (86,)
        p = np.atleast_1d(np.asarray(rid_to_ptau[rid], dtype=np.float32))  # (1,)
        if cond_mode == "combined":
            c = np.concatenate([a, p]).astype(np.float32)              # (87,) = [atrophy(86) | ptau217(1)] — order MUST match CombinedConditioner
        elif cond_mode == "ptau217":
            c = p
        else:
            c = a
        pet.append(r["pet"]); mri.append(r["mri"]); atr.append(a)
        diag.append(DIAG[r["group"]]); cond.append(c)
        split.append(r["split"]); grp.append(r["group"])

    if no_val_split:
        split = ["train" if _sp == "val" else _sp for _sp in split]
    print(f"[phase6] fold {fold_idx}/{n_folds}  cond={cond_mode}  "
          f"matched={len(pet)} (dropped {dropped})", flush=True)
    print(f"[phase6] cond_vec dim = {len(cond[0])}", flush=True)
    for s in ("train", "val", "test"):
        sub = [g for g, sp in zip(grp, split) if sp == s]
        print(f"   {s:5s} n={len(sub):4d}  " +
              "  ".join(f"{k}={sub.count(k)}" for k in ("CN", "MCI", "AD")), flush=True)

    def _ds(sel):
        idx = [i for i, s in enumerate(split) if s == sel]
        return SpatialTauPETDataset(
            [pet[i] for i in idx], [mri[i] for i in idx],
            [atr[i] for i in idx], [diag[i] for i in idx],
            atlas_data, inv_atlas_affine,
            use_dk_mask=use_dk_mask, use_tissue=use_tissue,
            cond_mode="ptau217",                      # cond delivered via ptau slot
            ptau_vals=[cond[i] for i in idx],
        )

    train_ds, val_ds, test_ds = _ds("train"), _ds("val"), _ds("test")
    mk = lambda d, sh: DataLoader(d, batch_size=batch_size, shuffle=sh,
                                  num_workers=num_workers, drop_last=False)
    return (train_ds, val_ds, test_ds,
            mk(train_ds, True), mk(val_ds, False), mk(test_ds, False))
