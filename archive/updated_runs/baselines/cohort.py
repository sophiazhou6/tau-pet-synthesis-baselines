"""
Manifest-driven cohort: one CSV describes the subjects, their files and the fixed
held-out test set; folds are derived deterministically from it.

Manifest columns (see scripts/prepare_cohort.py, which writes it):
    RID     subject id (string / int)
    group   diagnosis group: CN / MCI / AD (anything else is kept but sorted last)
    split   'heldout_test'  never trained on, never used for model selection, fixed for every fold
            'dev'           in the pool (= everything that is not test); the validation subjects
                            are carved out of the pool by the rule below
            'train' / 'val' in the pool, with a FIXED validation set given by whoever made the split
            (aliases accepted: test/heldout/holdout, trainval/pool, training, valid/validation)
            A manifest uses either 'dev' or 'train'/'val' for its pool, not both.
    fold    optional, pool rows only: an externally supplied CV fold (0..K-1 or 1..K)
    mri     absolute path to T1_to_MNI_nonlin.nii.gz
    pet     absolute path to PET_MNISpace_SUVR_CerebellumNorm.nii(.gz)

Policy (identical to TauGenNet's mentor split, src/splits.py, when the pool is 'dev'):
  * test = every 'heldout_test' row, in manifest order (this order is the subject_XXX
    index of cached generations and of every per-subject metric table);
  * fold_idx=i → validation = fold i of the pool: the manifest's `fold` column if present,
    otherwise StratifiedKFold(shuffle=True, random_state=seed) over the pool RIDs sorted
    numerically, stratified by group (train+val merged when the pool is 'train'/'val');
  * fold_idx=None → the manifest's fixed 'val' rows if there are any; otherwise a stratified
    val_frac of the pool — with val_frac=0.2 that is exactly CV fold 0; val_frac=0 → no
    validation, train on the whole pool.

Volumes are loaded by TauGenNet's own TauPETDataset (src/dataset_final.py): trilinear
resample to VOL_SHAPE, DK86 atlas mask, per-subject min-max to [0,1], with
get_pet_norms(i) for the SUVR inverse — so every baseline sees exactly the inputs the
diffusion model and the Dense-U-Net saw.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import SEED  # noqa: F401  (also puts TAUGENNET_ROOT on sys.path)
from src.dataset_final import TauPETDataset  # noqa: E402

GROUP_ORDER = ["CN", "MCI", "AD"]
REQUIRED_COLS = ["RID", "group", "split", "mri", "pet"]
POOL_SPLITS = ("dev", "train", "val")
SPLIT_ALIASES = {
    "heldout_test": "heldout_test", "test": "heldout_test", "heldout": "heldout_test",
    "holdout": "heldout_test", "hold_out": "heldout_test", "held_out": "heldout_test",
    "dev": "dev", "train_val": "dev", "trainval": "dev", "pool": "dev",
    "train": "train", "training": "train", "tr": "train",
    "val": "val", "valid": "val", "validation": "val",
}


def norm_split(v):
    """Canonical split name, or None if the value is not recognised."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return SPLIT_ALIASES.get(str(v).strip().lower().replace("-", "_").replace(" ", "_"))


def norm_rid(v) -> str:
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v).strip()


def read_manifest(path, check_files=True):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing columns {missing}; has {list(df.columns)}")
    df = df.copy()
    df["RID"] = df["RID"].map(norm_rid)
    df["group"] = df["group"].astype(str).str.strip().str.upper()
    raw = df["split"].copy()
    df["split"] = df["split"].map(norm_split)
    bad = sorted(set(raw[df["split"].isna()].astype(str)))
    if bad:
        raise ValueError(f"Unrecognised split values {bad}; use heldout_test / dev / train / val")
    if (df["split"] == "dev").any() and df["split"].isin(["train", "val"]).any():
        raise ValueError("Manifest mixes 'dev' with 'train'/'val' rows: either every pool subject is 'dev' "
                         "(validation derived by the fold rule) or every one is 'train'/'val' (fixed validation).")
    dup = df["RID"][df["RID"].duplicated()].tolist()
    if dup:
        raise ValueError(f"Duplicate RIDs in manifest: {dup[:10]}")
    if check_files:
        absent = [(r.RID, p) for r in df.itertuples() for p in (r.mri, r.pet) if not os.path.exists(p)]
        if absent:
            raise FileNotFoundError(f"{len(absent)} manifest files missing, e.g. {absent[:3]}")
    return df.reset_index(drop=True)


def class_map(groups):
    """group name → class index, CN < MCI < AD, then any other group alphabetically."""
    present = sorted(set(groups), key=lambda g: (GROUP_ORDER.index(g) if g in GROUP_ORDER else 99, g))
    return {g: i for i, g in enumerate(present)}


def stratified_val_folds(dev_df, n_folds, seed):
    """List[n_folds] of validation RID sets, StratifiedKFold(shuffle, seed) over sorted RIDs."""
    from sklearn.model_selection import StratifiedKFold
    rids = sorted(dev_df["RID"], key=lambda r: (len(r), r))
    grp = dict(zip(dev_df["RID"], dev_df["group"]))
    groups = [grp[r] for r in rids]
    counts = {g: groups.count(g) for g in set(groups)}
    if min(counts.values()) < n_folds:
        raise ValueError(f"StratifiedKFold needs >= {n_folds} subjects per group; have {counts}")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [{rids[i] for i in val_idx} for _, val_idx in skf.split(rids, groups)]


def _external_fold_val(pool, fold_idx, n_folds):
    """Validation RIDs of fold `fold_idx` from the manifest's `fold` column (0- or 1-based labels)."""
    f = pool["fold"]
    if f.isna().any():
        raise ValueError(f"`fold` column is missing for {int(f.isna().sum())} pool subjects")
    labels = sorted(set(int(v) for v in f))
    if labels == list(range(n_folds)):
        want = fold_idx
    elif labels == list(range(1, n_folds + 1)):
        want = fold_idx + 1
    else:
        raise ValueError(f"`fold` column has labels {labels}; expected 0..{n_folds - 1} or 1..{n_folds}")
    return set(pool.loc[f.astype(int) == want, "RID"])


def fold_indices(df, fold_idx=None, n_folds=5, seed=SEED, val_frac=0.2, verbose=True):
    """Positional (train_idx, val_idx, test_idx) into the manifest rows (policy: module docstring)."""
    test_idx = [i for i, s in enumerate(df["split"]) if s == "heldout_test"]
    pool = df[df["split"].isin(POOL_SPLITS)]
    fixed_val = set(pool.loc[pool["split"] == "val", "RID"])
    has_fold_col = "fold" in df.columns and pool["fold"].notna().any()
    if fold_idx is not None:
        if not (0 <= fold_idx < n_folds):
            raise ValueError(f"fold_idx {fold_idx} out of range [0,{n_folds})")
        if has_fold_col:
            val_rids, how = _external_fold_val(pool, fold_idx, n_folds), "fold column of the manifest"
        else:
            val_rids = stratified_val_folds(pool, n_folds, seed)[fold_idx]
            how = "stratified k-fold over the pool" + (" (fixed train/val merged)" if fixed_val else "")
    elif fixed_val:
        val_rids, how = fixed_val, "fixed validation set of the manifest"
    elif val_frac and val_frac > 0:
        k = max(2, int(round(1.0 / val_frac)))
        val_rids, how = stratified_val_folds(pool, k, seed)[0], f"stratified 1/{k} of the pool (= CV fold 0 when k={n_folds})"
    else:
        val_rids, how = set(), "no validation (whole pool is training)"
    train_idx = [i for i in pool.index if df.at[i, "RID"] not in val_rids]
    val_idx = [i for i in pool.index if df.at[i, "RID"] in val_rids]
    if verbose:
        tag = f"fold {fold_idx}/{n_folds}" if fold_idx is not None else "single split"
        print(f"[cohort] {tag}: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test "
              f"(pool {len(pool)}; val = {how}; groups {dict(df['group'].value_counts())})", flush=True)
    return train_idx, val_idx, test_idx


class CohortDataset(TauPETDataset):
    """TauPETDataset + subject metadata. cond = class index of the diagnosis group, shape (1,)."""

    def __init__(self, rows: pd.DataFrame, cls_map, use_dk_mask=True):
        rows = rows.reset_index(drop=True)
        labels = [np.array([float(cls_map[g])], dtype=np.float32) for g in rows["group"]]
        super().__init__(list(rows["pet"]), list(rows["mri"]), labels, use_dk_mask=use_dk_mask)
        self.rids = list(rows["RID"])
        self.groups = list(rows["group"])
        self.labels = [int(cls_map[g]) for g in rows["group"]]
        self.cls_map = dict(cls_map)

    def index_table(self):
        return pd.DataFrame({"index": range(len(self)), "RID": self.rids, "group": self.groups,
                             "label": self.labels, "pet": self.pet_paths, "mri": self.mri_paths})


def build_cohort(manifest, fold_idx=None, n_folds=5, seed=SEED, val_frac=0.2, batch_size=2,
                 use_dk_mask=True, num_workers=2, check_files=True, limit_subjects=None):
    """Returns dict(train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, cls_map, df).

    limit_subjects: debugging aid — keep only the first N rows of each of train/val/test.
    """
    df = read_manifest(manifest, check_files=check_files)
    cls_map = class_map(df["group"])
    tr, va, te = fold_indices(df, fold_idx, n_folds, seed, val_frac)
    if limit_subjects:
        tr, va, te = tr[:limit_subjects], va[:max(1, limit_subjects // 2)], te[:limit_subjects]
        print(f"[cohort] DEBUG limit_subjects={limit_subjects}: {len(tr)}/{len(va)}/{len(te)}")
    mk = lambda idx: CohortDataset(df.iloc[idx], cls_map, use_dk_mask=use_dk_mask)  # noqa: E731
    train_ds, val_ds, test_ds = mk(tr), mk(va), mk(te)
    kw = dict(num_workers=num_workers, pin_memory=True)
    return dict(
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **kw),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
        cls_map=cls_map, df=df)


class CachedVolumes(Dataset):
    """Materialise a CohortDataset once in RAM: (pet, mri, cond) tensors exactly as the
    dataset returns them (masked, [0,1]). ~8 MB per subject at VOL_SHAPE; 300 subjects
    ≈ 2.5 GB. Removes the per-epoch NIfTI reload + trilinear resample cost."""

    def __init__(self, ds, verbose=True):
        self.ds = ds
        self.items = []
        for i in range(len(ds)):
            pet, mri, cond = ds[i]
            self.items.append((pet.clone(), mri.clone(), cond.clone()))
            if verbose and (i + 1) % 50 == 0:
                print(f"  cached {i + 1}/{len(ds)} volumes", flush=True)
        self.labels = getattr(ds, "labels", [int(c.item()) for _, _, c in self.items])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class UnpairedView(Dataset):
    """For unpaired (CycleGAN-style) training: item i yields (mri_i, pet_j, pet_i, label_i)
    with j drawn at random every access, so MRI and PET come from different subjects
    (the ShareGAN training regime) while the paired PET stays available for the optional
    supervised term and for validation."""

    def __init__(self, cached: CachedVolumes, seed=SEED):
        self.c = cached
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        j = int(self.rng.integers(len(self.c)))
        pet_i, mri_i, cond_i = self.c[i]
        pet_j, _, _ = self.c[j]
        return mri_i, pet_j, pet_i, cond_i
