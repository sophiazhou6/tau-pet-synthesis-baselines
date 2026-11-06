#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
splits.py – mentor-provided subject split for 5-fold CV + held-out test.

Anil (mentor) fixed the exact subject split via two CSVs in data/raw:
    heldout_test_split.csv   47 subjects, column split == "heldout_test"
    train_val_split.csv     188 subjects, column split == "dev"  (the CV pool)
Both have columns: RID, group (AD/MCI), split.  The two sets are disjoint.

Policy (agreed with the user):
  * The held-out TEST set is exactly the 47 heldout_test RIDs — never trained on.
  * The 5 CV folds are built ONLY from the 188 dev RIDs, stratified by group
    (AD/MCI) with a fixed seed → deterministic, reproducible, balanced folds.
  * Any subject present on disk but in NEITHER CSV is DROPPED (use the exact
    split, not the extra 14 subjects that exist locally).

The fold partition itself is not specified by the mentor's files (no fold
column), so it is generated here with sklearn StratifiedKFold(shuffle=True,
random_state=seed). If Anil later provides an explicit fold column, swap
``stratified_kfold_val_sets`` for a lookup — the rest is unaffected.

Every dataset_*.py build_dataloaders should route its split through
``assign_indices`` so the test set is identical across all models.
"""

import csv
import os

TEST_CSV = os.environ.get("MENTOR_TEST_CSV", "heldout_test_split.csv")
DEV_CSV  = os.environ.get("MENTOR_DEV_CSV", "train_val_split.csv")


def _norm_rid(value) -> str:
    """Canonical RID string: '6264' (matches dataset_*.py rid keys)."""
    return str(int(float(value)))


def load_mentor_splits(data_dir, test_csv=TEST_CSV, dev_csv=DEV_CSV):
    """Return (test_dict, dev_dict): rid -> group ('AD'/'MCI').

    Raises if either file is missing or the two sets overlap.
    """
    def _load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Mentor split file not found: {path}\n"
                "Copy heldout_test_split.csv and train_val_split.csv into data/raw/."
            )
        out = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                out[_norm_rid(row["RID"])] = row.get("group", "")
        return out

    test_dict = _load(os.path.join(data_dir, test_csv))
    dev_dict  = _load(os.path.join(data_dir, dev_csv))
    overlap = set(test_dict) & set(dev_dict)
    if overlap:
        raise ValueError(f"RIDs appear in BOTH split files: {sorted(overlap)}")
    return test_dict, dev_dict


def stratified_kfold_val_sets(dev_rid_to_group, n_folds, seed):
    """List[n_folds] of sets — the validation RIDs for each fold.

    Stratified by group, deterministic (StratifiedKFold shuffle + fixed seed).
    RIDs are sorted first so the result depends only on (membership, seed).
    """
    from sklearn.model_selection import StratifiedKFold

    rids   = sorted(dev_rid_to_group, key=lambda r: int(r))
    groups = [dev_rid_to_group[r] for r in rids]
    min_class = min(groups.count(g) for g in set(groups)) if groups else 0
    if min_class < n_folds:
        raise ValueError(
            f"StratifiedKFold needs >= n_folds={n_folds} subjects per group, "
            f"but smallest group has {min_class}. Reduce n_folds or check the split."
        )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [{rids[i] for i in val_idx} for _, val_idx in skf.split(rids, groups)]


def assign_indices(rids, *, fold_idx, n_folds, seed, val_frac_of_dev,
                   test_dict, dev_dict, verbose=True):
    """Map an aligned list of subject RIDs to (train_idx, val_idx, test_idx).

    Parameters
    ----------
    rids            : list[str] — RID per matched subject, aligned with the
                      dataset's pet_paths/mri_paths/... lists (already filtered
                      to the mentor split; see ``keep_set``).
    fold_idx        : int or None. If given, val = that stratified CV fold of the
                      dev pool; train = remaining dev. If None, a single
                      stratified split is used (val ~ val_frac_of_dev of dev).
    n_folds         : number of CV folds (5).
    seed            : fixed RNG seed for the stratified partition.
    val_frac_of_dev : validation fraction of the dev pool when fold_idx is None.
    test_dict/dev_dict : rid -> group, from load_mentor_splits().

    Returns
    -------
    (train_idx, val_idx, test_idx) : lists of positional indices into ``rids``.
    """
    test_set = set(test_dict)
    dev_set  = set(dev_dict)
    present_dev = {r: dev_dict[r] for r in rids if r in dev_set}

    if fold_idx is not None:
        if not (0 <= fold_idx < n_folds):
            raise ValueError(f"fold_idx {fold_idx} out of range [0,{n_folds})")
        val_rids = stratified_kfold_val_sets(present_dev, n_folds, seed)[fold_idx]
    elif val_frac_of_dev and val_frac_of_dev > 0:
        k = max(2, round(1.0 / val_frac_of_dev))
        val_rids = stratified_kfold_val_sets(present_dev, k, seed)[0]
    else:
        val_rids = set()  # val_frac_of_dev == 0 (falsy) -> no held-out validation, bug fix 2026-07-23

    train_idx, val_idx, test_idx = [], [], []
    for i, r in enumerate(rids):
        if r in test_set:
            test_idx.append(i)
        elif r in val_rids:
            val_idx.append(i)
        elif r in dev_set:
            train_idx.append(i)
        # else: not in either split → already filtered out, ignore defensively

    if verbose:
        fold_tag = f"fold {fold_idx}/{n_folds}" if fold_idx is not None else "single split"
        print(f"[mentor split] {fold_tag}: "
              f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test "
              f"(dev pool present: {len(present_dev)})")
    return train_idx, val_idx, test_idx


def keep_set(test_dict, dev_dict):
    """Set of RIDs that belong to the mentor split (test ∪ dev).

    A subject whose RID is not in this set is dropped during matching.
    """
    return set(test_dict) | set(dev_dict)
