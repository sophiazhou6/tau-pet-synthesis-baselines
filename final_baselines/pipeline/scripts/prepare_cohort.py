#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Build a cohort manifest (and, optionally, a TauGenNet-compatible data/raw layout) from
imaging folders + a datasheet — the ingest step for the expanded cohort.

Two sources:

A) the lab layout already on disk ($TAUGENNET_ROOT/data/raw, with the mentor split CSVs):
     python scripts/prepare_cohort.py --from-lab-layout --name mentor235

B) freshly downloaded folders (the Box "MRI" and "Tau PET" folders) + the datasheet:
     python scripts/prepare_cohort.py --name cohort_v2 \
         --mri-dir /path/to/MRI --pet-dir /path/to/TauPET --datasheet /path/to/datasheet.csv \
         <split source, see below> --materialize $TAUGENNET_ROOT/data/raw_v2

   * MRI files are found by --mri-glob (default '**/*T1_to_MNI_nonlin*.nii*'), PET files by
     --pet-glob (default '**/*PET*SUVR*Cereb*.nii*'); the RID is parsed from the path with
     --rid-regex (default tries RID_<n>, <site>_S_<n>, then a bare number).
   * the datasheet provides the diagnosis group (column group/DX/Diagnosis…, normalised to
     CN/MCI/AD) and may carry 'split' (and 'fold') columns.

Split sources (any combination; later ones override earlier ones, overlaps between the
per-split files are an error):
   1. the datasheet's 'split' column;
   2. --split-csv FILE: one table with RID + split (+ optional group, fold) columns;
   3. per-split files, each a list of RIDs (+ optional group):
        --test-from  → heldout_test        --dev-from → dev (pool; validation derived by the fold rule)
        --train-from → train               --val-from → val (fixed validation set)
   Split values accepted: heldout_test/test/holdout, dev/trainval/pool, train, val/validation.
   A cohort's pool is either all 'dev' or all 'train'/'val'.
   Subjects that no source mentions are DROPPED (the summer policy for Anil's split) unless
   --unsplit assign, which splits them stratified by group: --grow-test-frac to heldout_test,
   the rest to the pool. With no split source at all, --unsplit assign gives a provisional
   stratified split of the whole cohort.

Subjects without a regional atrophy z-score vector (--atrophy-csv, default the lab's
regional_atrophy_zscores.csv; same rule as TauGenNet's loader: any row whose 86 *_ATROPHY_Z
values are not all NaN) are dropped (--no-require-atrophy to keep them). TauGenNet's loader and
evaluator drop them anyway, so keeping them would break the shared test set and the folds.

   * --materialize DIR writes a data/raw-compatible tree of symlinks
     (1mm_parcellated_{CN,MCI,AD}_subj/RID_<n>/T1_to_MNI_nonlin.nii.gz,
      cerebellumNormalized_AD_MCI/{CN,MCI,AD}/RID_<n>/PET_MNISpace_SUVR_CerebellumNorm.nii.gz,
      heldout_test_split.csv + train_val_split.csv (pool) in the mentor format and in the
      controls_322/ numeric format, fixed_val_split.csv when there is a fixed validation set,
      the atrophy CSV used above, and links to the atlas/other support files) so TauGenNet's
      own loaders can train on the same cohort.

Output: $BASELINES_MANIFEST_DIR/<name>.csv (+ <name>_report.txt). Never overwrites an
existing manifest unless --overwrite.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from denseunet.config import MANIFEST_DIR, SEED, TAUGENNET_ROOT  # noqa: E402
from denseunet.cohort import GROUP_ORDER, POOL_SPLITS, norm_rid, norm_split  # noqa: E402

RAW = os.path.join(TAUGENNET_ROOT, "data", "raw")
GROUP_ALIASES = {"CN": "CN", "NC": "CN", "NL": "CN", "HC": "CN", "NORMAL": "CN", "CONTROL": "CN", "CONTROLS": "CN",
                 "1": "CN", "1.0": "CN", "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI", "2": "MCI", "2.0": "MCI",
                 "AD": "AD", "DEMENTIA": "AD", "3": "AD", "3.0": "AD"}
SUPPORT_FILES = ["mni_dk_atlas.nii.gz", "dk86.lut", "regional_atrophy_zscores.csv",
                 "ADNI34Tau_withFluidBiomarkers.csv", "RegionLabelMap.py"]
RID_PATTERNS = [r"RID_?(\d+)", r"\d{3}_S_(\d+)", r"(?<!\d)(\d{3,5})(?!\d)"]


def norm_group(v):
    s = str(v).strip().upper()
    return GROUP_ALIASES.get(s, s)


def parse_rid(path, regex=None):
    pats = [regex] if regex else RID_PATTERNS
    rel = path
    for pat in pats:
        # search from the deepest path component outwards so subject folders win over root paths
        for comp in reversed(rel.split(os.sep)):
            m = re.search(pat, comp)
            if m:
                return norm_rid(m.group(1))
    return None


def scan(dir_, pattern, regex):
    found = {}
    files = sorted(glob.glob(os.path.join(dir_, pattern), recursive=True))
    dups = []
    for f in files:
        rid = parse_rid(os.path.relpath(f, dir_), regex)
        if rid is None:
            continue
        if rid in found:
            dups.append((rid, f))
            continue
        found[rid] = os.path.abspath(f)
    return found, files, dups


def _read_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(path)
        except ImportError:
            sys.exit(f"{path}: reading Excel needs the `openpyxl` package (not in the taugennet env). "
                     "Either `pip install openpyxl` or save it as CSV and pass that.")
    return pd.read_csv(path)


def _col(df, names):
    cols = {c.lower().strip(): c for c in df.columns}
    return next((cols[n] for n in names if n in cols), None)


def _rid_of(v):
    s = str(v).strip()
    return norm_rid(s) if re.fullmatch(r"\d+(\.0+)?", s) else parse_rid(s)


def read_table_columns(path, want_split=False, rid_col=None, group_col=None, require_rid=True):
    """{rid: group}, {rid: split}, {rid: fold} from a datasheet / split table / RID list.
    Unknown split values abort with the list of offending values. With require_rid=False a
    table without a RID column (e.g. a bare list of RIDs) yields empty dicts."""
    df = _read_table(path)
    rid_col = rid_col or _col(df, ("rid", "subject", "subject_id", "ptid", "id"))
    if rid_col is None:
        if not require_rid:
            return {}, {}, {}
        sys.exit(f"{path}: no RID column (columns {list(df.columns)}); pass --rid-col")
    group_col = group_col or _col(df, ("group", "dx", "diagnosis", "dx_bl", "dx_group", "cohort", "label"))
    split_col = _col(df, ("split", "set", "partition", "subset")) if want_split else None
    fold_col = _col(df, ("fold", "cv_fold", "fold_idx")) if want_split else None
    groups, split, fold, bad = {}, {}, {}, set()
    for _, row in df.iterrows():
        rid = _rid_of(row[rid_col])
        if rid is None:
            continue
        if group_col is not None and pd.notna(row[group_col]):
            groups[rid] = norm_group(row[group_col])
        if split_col is not None and pd.notna(row[split_col]):
            sp = norm_split(row[split_col])
            if sp is None:
                bad.add(str(row[split_col]))
            else:
                split[rid] = sp
        if fold_col is not None and pd.notna(row[fold_col]):
            fold[rid] = int(float(row[fold_col]))
    if bad:
        sys.exit(f"{path}: unrecognised split values {sorted(bad)} (use heldout_test/test, dev/trainval, train, val)")
    return groups, split, fold


def _read_rid_list(path):
    """RIDs of a split file that may be a bare list (one RID per line, with or without header)."""
    try:
        df = _read_table(path)
    except Exception:
        return []
    col = _col(df, ("rid", "subject", "subject_id", "ptid", "id"))
    vals = df[col] if col is not None else pd.concat([pd.Series([df.columns[0]]), df.iloc[:, 0]])
    return [r for r in (_rid_of(v) for v in vals if pd.notna(v)) if r is not None]


def atrophy_vectors(path):
    """rid → 86-vector, exactly TauGenNet's _build_rid_to_atrophy: rows whose 86 *_ATROPHY_Z
    values are all NaN are skipped, later rows win (longitudinal duplicates)."""
    from src.dataset_final import REGION_COLS  # noqa: E402
    df = _read_table(path)
    missing = [c for c in REGION_COLS if c not in df.columns]
    if missing:
        sys.exit(f"{path}: atrophy table lacks {len(missing)} of the 86 *_ATROPHY_Z columns, e.g. {missing[:3]}")
    rid_col = _col(df, ("rid",)) or "RID"
    vals = df[REGION_COLS].to_numpy(dtype=np.float64)
    out = {}
    for r, v in zip(df[rid_col], vals):
        if pd.isna(r) or np.all(np.isnan(v)):
            continue
        out[norm_rid(r)] = v
    return out


def stratified_split(rids, groups, frac, seed):
    """Return the set of RIDs assigned to the test side (fraction frac, per group, seeded)."""
    rng = np.random.default_rng(seed)
    test = set()
    for g in sorted(set(groups)):
        members = sorted([r for r, gg in zip(rids, groups) if gg == g], key=lambda r: (len(r), r))
        rng.shuffle(members)
        k = int(round(frac * len(members)))
        test.update(members[:k])
    return test


def materialize(df, out_dir, atrophy_csv=None):
    os.makedirs(out_dir, exist_ok=True)
    for r in df.itertuples():
        g = r.group if r.group in GROUP_ORDER else "OTHER"
        md = os.path.join(out_dir, f"1mm_parcellated_{g}_subj", f"RID_{r.RID}")
        pd_ = os.path.join(out_dir, "cerebellumNormalized_AD_MCI", g, f"RID_{r.RID}")
        os.makedirs(md, exist_ok=True); os.makedirs(pd_, exist_ok=True)
        for src, dst in ((r.mri, os.path.join(md, "T1_to_MNI_nonlin.nii.gz")),
                         (r.pet, os.path.join(pd_, "PET_MNISpace_SUVR_CerebellumNorm" + (".nii.gz" if r.pet.endswith(".gz") else ".nii")))):
            if not os.path.lexists(dst):
                os.symlink(src, dst)
        seg = r.mri.replace("T1_to_MNI_nonlin.nii.gz", "T1_seg_in_MNI.nii.gz")
        if os.path.exists(seg) and not os.path.lexists(os.path.join(md, "T1_seg_in_MNI.nii.gz")):
            os.symlink(seg, os.path.join(md, "T1_seg_in_MNI.nii.gz"))
    test_df = df[df.split == "heldout_test"][["RID", "group"]].assign(split="heldout_test")
    pool_df = df[df.split.isin(POOL_SPLITS)][["RID", "group"]].assign(split="dev")   # TauGenNet: the CV pool
    test_df.to_csv(os.path.join(out_dir, "heldout_test_split.csv"), index=False)
    pool_df.to_csv(os.path.join(out_dir, "train_val_split.csv"), index=False)
    if (df.split == "val").any():   # TauGenNet's splits.py derives val itself; kept for reference / a patch
        df[df.split == "val"][["RID", "group"]].to_csv(os.path.join(out_dir, "fixed_val_split.csv"), index=False)
    if atrophy_csv:
        dst = os.path.join(out_dir, "regional_atrophy_zscores.csv")
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(atrophy_csv), dst)
    # TauGenNet's `--use-controls-322 --controls-dir` path wants 2-col CSVs with numeric groups (1=CN, 2=MCI, 3=AD)
    c322 = os.path.join(out_dir, "controls_322"); os.makedirs(c322, exist_ok=True)
    code = {"CN": 1, "MCI": 2, "AD": 3}
    for sub_df, fname in ((test_df, "heldout_test_split.csv"), (pool_df, "train_val_split.csv")):
        sub = sub_df[["RID", "group"]].copy()
        sub["group"] = sub["group"].map(lambda g: float(code.get(g, 0)))
        sub.to_csv(os.path.join(c322, fname), index=False)
    for f in SUPPORT_FILES:
        src, dst = os.path.join(RAW, f), os.path.join(out_dir, f)
        if f == "regional_atrophy_zscores.csv" and atrophy_csv:
            continue
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    print(f"[materialize] TauGenNet-compatible layout → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--from-lab-layout", action="store_true")
    p.add_argument("--raw-dir", default=RAW)
    p.add_argument("--mri-dir"); p.add_argument("--pet-dir"); p.add_argument("--datasheet")
    p.add_argument("--mri-glob", default="**/*T1_to_MNI_nonlin*.nii*")
    p.add_argument("--pet-glob", default="**/*PET*SUVR*Cereb*.nii*")
    p.add_argument("--rid-regex", default=None); p.add_argument("--rid-col"); p.add_argument("--group-col")
    p.add_argument("--split-csv", default=None, help="one table with RID + split (+ group, fold)")
    p.add_argument("--test-from", default=None); p.add_argument("--dev-from", default=None)
    p.add_argument("--train-from", default=None); p.add_argument("--val-from", default=None)
    p.add_argument("--group-source", choices=["split", "datasheet"], default="split",
                   help="when both the split files and the datasheet give a diagnosis group: which one is used "
                        "(default: the split files, which is what the split was stratified on)")
    p.add_argument("--unsplit", choices=["drop", "assign"], default="drop",
                   help="subjects no split source mentions: drop (default) or split them stratified")
    p.add_argument("--grow-test-frac", type=float, default=0.2, help="--unsplit assign: fraction to heldout_test")
    p.add_argument("--atrophy-csv", default=os.path.join(RAW, "regional_atrophy_zscores.csv"))
    p.add_argument("--require-atrophy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--materialize", default=None)
    p.add_argument("--out-dir", default=MANIFEST_DIR)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out_csv = os.path.join(a.out_dir, a.name + ".csv")
    if os.path.exists(out_csv) and not a.overwrite:
        sys.exit(f"{out_csv} exists (pass --overwrite)")
    report = []
    split, fold = {}, {}

    if a.from_lab_layout:
        mri = {}
        for f in glob.glob(os.path.join(a.raw_dir, "1mm_parcellated_*_subj", "*", "T1_to_MNI_nonlin.nii.gz")):
            mri[norm_rid(os.path.basename(os.path.dirname(f)).split("_")[-1])] = f
        pet, groups = {}, {}
        for f in glob.glob(os.path.join(a.raw_dir, "cerebellumNormalized_AD_MCI", "*", "RID_*", "PET_MNISpace_SUVR_CerebellumNorm.nii*")):
            rid = norm_rid(os.path.basename(os.path.dirname(f)).replace("RID_", ""))
            pet[rid] = f; groups[rid] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        a.test_from = a.test_from or os.path.join(a.raw_dir, "heldout_test_split.csv")
        a.dev_from = a.dev_from or os.path.join(a.raw_dir, "train_val_split.csv")
        report.append(f"lab layout {a.raw_dir}: {len(mri)} MRI, {len(pet)} PET")
    else:
        if not (a.mri_dir and a.pet_dir):
            sys.exit("need --mri-dir and --pet-dir (or --from-lab-layout)")
        mri, mri_files, mdup = scan(a.mri_dir, a.mri_glob, a.rid_regex)
        pet, pet_files, pdup = scan(a.pet_dir, a.pet_glob, a.rid_regex)
        report.append(f"MRI dir {a.mri_dir}: {len(mri_files)} files matched '{a.mri_glob}', {len(mri)} RIDs, {len(mdup)} duplicates")
        report.append(f"PET dir {a.pet_dir}: {len(pet_files)} files matched '{a.pet_glob}', {len(pet)} RIDs, {len(pdup)} duplicates")
        for rid, f in (mdup + pdup)[:20]:
            report.append(f"   duplicate RID {rid}: {f} (first occurrence kept)")
        groups = {}
        if a.datasheet:      # optional: the split files may carry the diagnosis group themselves
            groups, sheet_split, sheet_fold = read_table_columns(a.datasheet, True, a.rid_col, a.group_col)
            split.update(sheet_split); fold.update(sheet_fold)
            report.append(f"datasheet {a.datasheet}: {len(groups)} RIDs with a group, {len(sheet_split)} with a split, "
                          f"{len(sheet_fold)} with a fold")

    # split sources: datasheet < --split-csv < per-split files
    if a.split_csv:
        g2, s2, f2 = read_table_columns(a.split_csv, True, a.rid_col)
        if not s2:
            sys.exit(f"{a.split_csv}: no split column found")
        changed = sum(1 for r, v in s2.items() if r in split and split[r] != v)
        split.update(s2); fold.update(f2)
        for r, g in g2.items():
            if a.group_source == "split":
                groups[r] = g
            else:
                groups.setdefault(r, g)
        report.append(f"split table {a.split_csv}: {len(s2)} RIDs ({changed} override the datasheet), {len(f2)} with a fold")
    per_file = {}
    for path, name in ((a.dev_from, "dev"), (a.train_from, "train"), (a.val_from, "val"), (a.test_from, "heldout_test")):
        if path:
            g2, _, _ = read_table_columns(path, False, a.rid_col, require_rid=False)
            per_file[name] = set(g2) | set(_read_rid_list(path))
            if not per_file[name]:
                sys.exit(f"{path}: no RIDs found")
            for r, g in g2.items():
                if a.group_source == "split":
                    groups[r] = g
                else:
                    groups.setdefault(r, g)
            report.append(f"{name:12s} ← {path}: {len(per_file[name])} RIDs")
    names = list(per_file)
    clashes = [(x, y, sorted(per_file[x] & per_file[y])) for i, x in enumerate(names) for y in names[i + 1:]
               if per_file[x] & per_file[y]]
    if clashes:
        sys.exit("RIDs listed in more than one split file:\n" + "\n".join(f"  {x} ∩ {y}: {v[:20]}" for x, y, v in clashes))
    for name, rids in per_file.items():
        for r in rids:
            split[r] = name
    pool_kinds = {v for v in split.values() if v in POOL_SPLITS}
    if "dev" in pool_kinds and pool_kinds & {"train", "val"}:
        n_dev = sum(v == "dev" for v in split.values())
        sys.exit(f"the split sources mix 'dev' ({n_dev} RIDs) with 'train'/'val': give either a pool "
                 "(dev, validation derived by the fold rule) or explicit train/val lists, not both")

    matched = sorted(set(mri) & set(pet) & set(groups), key=lambda r: (len(r), r))
    only_mri = sorted(set(mri) - set(pet)); only_pet = sorted(set(pet) - set(mri))
    no_group = sorted((set(mri) & set(pet)) - set(groups))
    report.append(f"matched (MRI ∧ PET ∧ group): {len(matched)}   MRI-only: {len(only_mri)}   PET-only: {len(only_pet)}   "
                  f"no group: {len(no_group)}")
    if only_mri: report.append(f"   MRI without PET: {only_mri[:30]}{' …' if len(only_mri) > 30 else ''}")
    if only_pet: report.append(f"   PET without MRI: {only_pet[:30]}{' …' if len(only_pet) > 30 else ''}")
    if no_group: report.append(f"   no group: {no_group[:30]}{' …' if len(no_group) > 30 else ''}")
    split_only = sorted(set(split) - set(matched), key=lambda r: (len(r), r))
    if split_only:
        report.append(f"   in the split but not matched (no MRI/PET/group): {len(split_only)} {split_only[:30]}"
                      f"{' …' if len(split_only) > 30 else ''}")

    if a.require_atrophy:
        atr = atrophy_vectors(a.atrophy_csv)
        no_atr = [r for r in matched if r not in atr]
        partial = [r for r in matched if r in atr and np.isnan(atr[r]).any()]
        matched = [r for r in matched if r in atr]
        report.append(f"atrophy z-scores {a.atrophy_csv}: dropped {len(no_atr)} subjects without a vector"
                      + (f" {no_atr[:30]}{' …' if len(no_atr) > 30 else ''}" if no_atr else "")
                      + (f"; {len(partial)} kept subjects have some regions NaN {partial[:10]}" if partial else ""))
        lost = [r for r in no_atr if split.get(r) == "heldout_test"]
        if lost:
            report.append(f"   NOTE: {len(lost)} of them were in the held-out test list: {lost[:20]}")

    unsplit = [r for r in matched if r not in split]
    if unsplit and a.unsplit == "assign":
        pool_name = "train" if pool_kinds & {"train", "val"} else "dev"
        new_test = stratified_split(unsplit, [groups[r] for r in unsplit], a.grow_test_frac, a.seed)
        for r in unsplit:
            split[r] = "heldout_test" if r in new_test else pool_name
        report.append(f"unsplit subjects: {len(unsplit)} → {len(new_test)} to heldout_test, "
                      f"{len(unsplit) - len(new_test)} to {pool_name} (stratified, frac {a.grow_test_frac}, seed {a.seed})")
    elif unsplit:
        report.append(f"dropped {len(unsplit)} subjects that no split source mentions (--unsplit assign to keep): "
                      f"{unsplit[:30]}{' …' if len(unsplit) > 30 else ''}")
    rows = [dict(RID=r, group=groups[r], split=split[r], mri=mri[r], pet=pet[r]) for r in matched if r in split]
    df = pd.DataFrame(rows)
    if df.empty:
        hint = "" if split else " (no split source given: pass the split, or --unsplit assign for a provisional one)"
        sys.exit("no subjects left" + hint + " — report so far:\n" + "\n".join(report))
    if fold:
        df["fold"] = [fold.get(r) if s in POOL_SPLITS else None for r, s in zip(df.RID, df.split)]
        miss = int(df.loc[df.split.isin(POOL_SPLITS), "fold"].isna().sum())
        if miss:
            report.append(f"WARNING: fold labels given for some pool subjects but missing for {miss}; "
                          "CV with the fold column will refuse to run until every pool subject has one")
        df["fold"] = df["fold"].astype("Int64")
    # Row order = TauGenNet's loader order (cohort dirs AD, MCI, CN; RID_* dirs string-sorted),
    # so subject_XXX indices of cached generations line up with evaluate_final's own test order.
    LAB_ORDER = ["AD", "MCI", "CN"]
    df["_o"] = df["group"].map(lambda g: LAB_ORDER.index(g) if g in LAB_ORDER else 99)
    df["_s"] = (df["split"] != "heldout_test").astype(int)
    df = df.sort_values(["_s", "_o", "RID"]).drop(columns=["_o", "_s"])
    df.to_csv(out_csv, index=False)
    report.append("\n" + pd.crosstab(df["group"], df["split"], margins=True).to_string())
    if "fold" in df.columns:
        report.append("\npool subjects per fold:\n" + pd.crosstab(df["group"], df["fold"]).to_string())
    for g in sorted(set(df.group)):
        if g not in GROUP_ORDER:
            report.append(f"WARNING: unrecognised group label '{g}'")
    txt = "\n".join(report)
    with open(os.path.join(a.out_dir, a.name + "_report.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n[manifest] {len(df)} subjects → {out_csv}")
    if a.materialize:
        materialize(df, a.materialize, a.atrophy_csv if a.require_atrophy else None)


if __name__ == "__main__":
    main()
