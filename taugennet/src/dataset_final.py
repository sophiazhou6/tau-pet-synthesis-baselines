"""
dataset_final.py – ADNI dataset loader supporting two conditioning modes.

Changes from dataset.py:
  - 80 / 10 / 10  train / val / test split (was 72 / 11 / 28).
  - _load() caches (orig_min, orig_max) per sample so callers can convert
    generated volumes back to SUVR space.  Use get_pet_norms(idx) +
    unnormalize() for evaluation in real space.

mode='ptau217'
    Conditioning data: scalar plasma p-tau217 (pg/mL), shape (1,).
    Subjects without a matched pT217_F value are excluded.
    Source: ADNI34Tau_withFluidBiomarkers.csv  column  pT217_F

mode='atrophy'
    Conditioning data: 86-dim regional atrophy z-score vector, shape (86,).
    Source: regional_atrophy_zscores.csv  columns CTX_LH_*/RIGHT_*/LEFT_*_ATROPHY_Z

Both modes:
    - Shuffle subjects before splitting so AD and MCI cohorts are mixed.
    - Volumes are normalised per-subject to [0, 1] and resampled to VOL_SHAPE.
    - Masked to Desikan-Killiany (DK) atlas (86 regions).
    - MRI files: data/{1mm_parcellated_AD_subj,1mm_parcellated_MCI_subj}/*/T1_to_MNI_nonlin.nii.gz
    - PET files: data/cerebellumNormalized_AD_MCI/{AD,MCI}/RID_*/PET_MNISpace_SUVR_CerebellumNorm.nii[.gz]
"""

import os
import glob
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from .config import BASE_DIR, VOL_SHAPE, BATCH_SIZE, ADNI_FLUID_CSV, SEED
from .atlas_labels import load_atlas, get_default_atlas_path
from .splits import load_mentor_splits, assign_indices, keep_set

# ── FreeSurfer atlas ID → column name prefix ──────────────────────────────────
_ATLAS = {
    1:  "CTX_LH_BANKSSTS",            2:  "CTX_LH_CAUDALANTERIORCINGULATE",
    3:  "CTX_LH_CAUDALMIDDLEFRONTAL",  4:  "CTX_LH_CUNEUS",
    5:  "CTX_LH_ENTORHINAL",           6:  "CTX_LH_FUSIFORM",
    7:  "CTX_LH_INFERIORPARIETAL",     8:  "CTX_LH_INFERIORTEMPORAL",
    9:  "CTX_LH_ISTHMUSCINGULATE",     10: "CTX_LH_LATERALOCCIPITAL",
    11: "CTX_LH_LATERALORBITOFRONTAL", 12: "CTX_LH_LINGUAL",
    13: "CTX_LH_MEDIALORBITOFRONTAL",  14: "CTX_LH_MIDDLETEMPORAL",
    15: "CTX_LH_PARAHIPPOCAMPAL",      16: "CTX_LH_PARACENTRAL",
    17: "CTX_LH_PARSOPERCULARIS",      18: "CTX_LH_PARSORBITALIS",
    19: "CTX_LH_PARSTRIANGULARIS",     20: "CTX_LH_PERICALCARINE",
    21: "CTX_LH_POSTCENTRAL",          22: "CTX_LH_POSTERIORCINGULATE",
    23: "CTX_LH_PRECENTRAL",           24: "CTX_LH_PRECUNEUS",
    25: "CTX_LH_ROSTRALANTERIORCINGULATE", 26: "CTX_LH_ROSTRALMIDDLEFRONTAL",
    27: "CTX_LH_SUPERIORFRONTAL",      28: "CTX_LH_SUPERIORPARIETAL",
    29: "CTX_LH_SUPERIORTEMPORAL",     30: "CTX_LH_SUPRAMARGINAL",
    31: "CTX_LH_FRONTALPOLE",          32: "CTX_LH_TEMPORALPOLE",
    33: "CTX_LH_TRANSVERSETEMPORAL",   34: "CTX_LH_INSULA",
    35: "CTX_RH_BANKSSTS",             36: "CTX_RH_CAUDALANTERIORCINGULATE",
    37: "CTX_RH_CAUDALMIDDLEFRONTAL",  38: "CTX_RH_CUNEUS",
    39: "CTX_RH_ENTORHINAL",           40: "CTX_RH_FUSIFORM",
    41: "CTX_RH_INFERIORPARIETAL",     42: "CTX_RH_INFERIORTEMPORAL",
    43: "CTX_RH_ISTHMUSCINGULATE",     44: "CTX_RH_LATERALOCCIPITAL",
    45: "CTX_RH_LATERALORBITOFRONTAL", 46: "CTX_RH_LINGUAL",
    47: "CTX_RH_MEDIALORBITOFRONTAL",  48: "CTX_RH_MIDDLETEMPORAL",
    49: "CTX_RH_PARAHIPPOCAMPAL",      50: "CTX_RH_PARACENTRAL",
    51: "CTX_RH_PARSOPERCULARIS",      52: "CTX_RH_PARSORBITALIS",
    53: "CTX_RH_PARSTRIANGULARIS",     54: "CTX_RH_PERICALCARINE",
    55: "CTX_RH_POSTCENTRAL",          56: "CTX_RH_POSTERIORCINGULATE",
    57: "CTX_RH_PRECENTRAL",           58: "CTX_RH_PRECUNEUS",
    59: "CTX_RH_ROSTRALANTERIORCINGULATE", 60: "CTX_RH_ROSTRALMIDDLEFRONTAL",
    61: "CTX_RH_SUPERIORFRONTAL",      62: "CTX_RH_SUPERIORPARIETAL",
    63: "CTX_RH_SUPERIORTEMPORAL",     64: "CTX_RH_SUPRAMARGINAL",
    65: "CTX_RH_FRONTALPOLE",          66: "CTX_RH_TEMPORALPOLE",
    67: "CTX_RH_TRANSVERSETEMPORAL",   68: "CTX_RH_INSULA",
    69: "LEFT_CEREBELLUM_CORTEX",      70: "LEFT_THALAMUS_PROPER",
    71: "LEFT_CAUDATE",                72: "LEFT_PUTAMEN",
    73: "LEFT_PALLIDUM",               74: "LEFT_HIPPOCAMPUS",
    75: "LEFT_AMYGDALA",               76: "LEFT_ACCUMBENS_AREA",
    77: "LEFT_VENTRALDC",              78: "RIGHT_CEREBELLUM_CORTEX",
    79: "RIGHT_THALAMUS_PROPER",       80: "RIGHT_CAUDATE",
    81: "RIGHT_PUTAMEN",               82: "RIGHT_PALLIDUM",
    83: "RIGHT_HIPPOCAMPUS",           84: "RIGHT_AMYGDALA",
    85: "RIGHT_ACCUMBENS_AREA",        86: "RIGHT_VENTRALDC",
}
REGION_COLS = [f"{_ATLAS[i]}_ATROPHY_Z" for i in range(1, 87)]


# ── Normalization helpers ─────────────────────────────────────────────────────

def unnormalize(vol_norm, orig_min, orig_max):
    """Invert per-subject min-max normalization to recover SUVR values."""
    return vol_norm * (orig_max - orig_min + 1e-8) + orig_min


# ── Dataset class ─────────────────────────────────────────────────────────────

class TauPETDataset(Dataset):
    """
    Returns (pet, mri, cond_data) where:
      pet, mri   – (1, H, W, D) float tensor normalised to [0, 1]
      cond_data  – (1,)  scalar p-tau217  [ptau217 mode]
                   (86,) atrophy z-scores [atrophy  mode]

    Call get_pet_norms(idx) to retrieve the (orig_min, orig_max) for subject
    idx, which can then be passed to unnormalize() to recover SUVR values.
    """

    def __init__(self, pet_paths, mri_paths, cond_values, use_dk_mask=True, suvr_values=None):
        self.pet_paths    = pet_paths
        self.mri_paths    = mri_paths
        self.cond_values  = cond_values  # list of np.ndarray or float
        self.suvr_values  = suvr_values  # list of (86,) np.ndarray or None, parallel to cond_values
        self.use_dk_mask  = use_dk_mask
        self._pet_norms   = {}          # idx -> (orig_min, orig_max), populated lazily
        self._dk_labels   = None
        # Precompute a single global DK86 atlas mask (and, if painting SUVR, the raw integer
        # label volume) at VOL_SHAPE resolution. All subjects are in MNI space, so one shared
        # mask/label-volume applies to all — same assumption the existing mask already relies on.
        if use_dk_mask or suvr_values is not None:
            atlas_data, _, _ = load_atlas(get_default_atlas_path())
            if use_dk_mask:
                binary = (atlas_data > 0).astype(np.float32)  # labels 1-86 → 1.0
                mask_t = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)  # (1,1,aH,aW,aD)
                self._dk_mask = F.interpolate(mask_t, size=VOL_SHAPE,
                                              mode="nearest").squeeze(0)      # (1,H,W,D)
            else:
                self._dk_mask = None
            if suvr_values is not None:
                lbl_t = torch.from_numpy(atlas_data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                self._dk_labels = F.interpolate(lbl_t, size=VOL_SHAPE,
                                                mode="nearest").squeeze(0).squeeze(0).long()  # (H,W,D)
        else:
            self._dk_mask = None

    def _get_dk_mask(self, idx):
        return self._dk_mask

    def _load(self, path, mask=None):
        import nibabel as nib
        vol = nib.load(path).get_fdata().astype(np.float32)
        vol = torch.tensor(vol).unsqueeze(0)
        vol = F.interpolate(
            vol.unsqueeze(0), size=VOL_SHAPE, mode="trilinear", align_corners=False
        ).squeeze(0)
        if mask is not None:
            vol = vol * mask
        orig_min = float(vol.min())
        orig_max = float(vol.max())
        return (vol - orig_min) / (orig_max - orig_min + 1e-8), orig_min, orig_max

    def get_pet_norms(self, idx):
        """Return (orig_min, orig_max) in SUVR for the PET volume at idx."""
        if idx not in self._pet_norms:
            import nibabel as nib
            vol = nib.load(self.pet_paths[idx]).get_fdata().astype(np.float32)
            self._pet_norms[idx] = (float(vol.min()), float(vol.max()))
        return self._pet_norms[idx]

    def __len__(self):
        return len(self.pet_paths)

    def __getitem__(self, idx):
        mask = self._get_dk_mask(idx) if self.use_dk_mask else None
        pet, pet_min, pet_max = self._load(self.pet_paths[idx], mask=mask)
        self._pet_norms[idx]  = (pet_min, pet_max)
        mri, _, _             = self._load(self.mri_paths[idx], mask=mask)
        cond = torch.tensor(self.cond_values[idx], dtype=torch.float32)

        if self.suvr_values is not None and self.suvr_values[idx] is not None:
            vec = torch.as_tensor(self.suvr_values[idx], dtype=torch.float32)  # (86,) or (K,86)
            if vec.ndim == 1:
                vec = vec.unsqueeze(0)                                         # -> (1,86)
            labels = self._dk_labels                                          # (H,W,D)
            fg = labels > 0
            chans = []
            for k in range(vec.shape[0]):
                painted = torch.zeros_like(labels, dtype=torch.float32)
                painted[fg] = vec[k][labels[fg] - 1]
                chans.append(painted.unsqueeze(0))
            mri = torch.cat([mri] + chans, dim=0)                              # (1+K,H,W,D)
        return pet, mri, cond


# ── Loaders ───────────────────────────────────────────────────────────────────

def _build_rid_to_mri(base_dir, extra_cohorts=None):
    rid_to_mri = {}
    cohorts = ["1mm_parcellated_AD_subj", "1mm_parcellated_MCI_subj"] + (extra_cohorts or [])
    for cohort in cohorts:
        for subj_dir in sorted(glob.glob(os.path.join(base_dir, cohort, "*"))):
            rid = os.path.basename(subj_dir).split("_")[-1]
            mri = os.path.join(subj_dir, "T1_to_MNI_nonlin.nii.gz")
            if os.path.exists(mri):
                rid_to_mri[rid] = mri
    return rid_to_mri


def _build_rid_to_suvr(base_dir):
    suvr_cols = [_ATLAS[i] for i in range(1, 87)]
    df = pd.read_csv(os.path.join(base_dir, "regional_SUVR_cerebellumNormalizedGTScan1.csv"))
    missing = [c for c in suvr_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Regional SUVR CSV missing columns: {missing}")
    rid_to_suvr = {}
    skipped = 0
    for _, row in df.iterrows():
        rid = str(int(row["RID"]))
        vec = row[suvr_cols].values.astype(np.float32)
        if np.all(np.isnan(vec)):
            skipped += 1
            continue
        rid_to_suvr[rid] = vec
    print(f"Regional SUVR vectors loaded: {len(rid_to_suvr)}  (skipped {skipped} all-NaN)")
    return rid_to_suvr


def _build_rid_to_atrophy(base_dir):
    df = pd.read_csv(os.path.join(base_dir, "regional_atrophy_zscores.csv"))
    missing = [c for c in REGION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Atrophy CSV missing columns: {missing}")
    rid_to_atrophy = {}
    skipped = 0
    duplicates = 0
    for _, row in df.iterrows():
        rid = str(int(row["RID"]))
        vec = row[REGION_COLS].values.astype(np.float32)
        if np.all(np.isnan(vec)):
            skipped += 1
            continue
        if rid in rid_to_atrophy:
            duplicates += 1
        rid_to_atrophy[rid] = vec
    if duplicates:
        print(f"Note: {duplicates} longitudinal duplicate rows in atrophy CSV — keeping latest visit per subject")
    if skipped:
        print(f"Skipped {skipped} rows with all-NaN atrophy vectors")
    return rid_to_atrophy


def _build_rid_to_ptau(fluid_csv):
    df = pd.read_csv(fluid_csv)
    return {
        str(int(row["RID"])): np.array([row["pT217_F"]], dtype=np.float32)
        for _, row in df.iterrows()
        if pd.notna(row["pT217_F"])
    }


def _load_controls_322_splits(controls_dir):
    """2-col (RID,group) format: which FILE a RID is in IS the split (no 'split' column,
    unlike load_mentor_splits' 3-col AD/MCI format). group: 1.0=CN, 2.0=MCI, 3.0=AD."""
    dev_df = pd.read_csv(os.path.join(controls_dir, "train_val_split.csv"))
    test_df = pd.read_csv(os.path.join(controls_dir, "heldout_test_split.csv"))
    dev_dict = {str(int(r["RID"])): int(r["group"]) for _, r in dev_df.iterrows()}
    test_dict = {str(int(r["RID"])): int(r["group"]) for _, r in test_df.iterrows()}
    return test_dict, dev_dict


def _build_controls_322_dataloaders(mode, base_dir, controls_dir, batch_size, seed,
                                     use_dk_mask, train_frac, val_frac, fold_idx, n_folds,
                                     suvr_input, suvr_as_cond):
    assert not (suvr_input and suvr_as_cond), "suvr_input and suvr_as_cond are mutually exclusive"
    if mode != "atrophy":
        raise ValueError(f"controls_322 cohort currently only supports mode='atrophy', got {mode!r}")

    rid_to_mri = _build_rid_to_mri(base_dir, extra_cohorts=["1mm_parcellated_CN_subj"])
    print(f"MRI subjects found (incl. CN): {len(rid_to_mri)}")

    rid_to_cond = _build_rid_to_atrophy(base_dir)
    print(f"Atrophy vectors loaded: {len(rid_to_cond)}")

    test_dict, dev_dict = _load_controls_322_splits(controls_dir)
    keep = keep_set(test_dict, dev_dict)
    print(f"controls_322 split: {len(test_dict)} heldout_test + {len(dev_dict)} dev "
          f"(subjects not in either file are dropped)")

    pet_paths, mri_paths, cond_vals, rids = [], [], [], []
    for cohort in ["AD", "MCI", "CN"]:
        cohort_dir = os.path.join(base_dir, "cerebellumNormalized_AD_MCI", cohort)
        if not os.path.exists(cohort_dir):
            print(f"Warning: {cohort_dir} not found, skipping")
            continue
        for subj_dir in sorted(glob.glob(os.path.join(cohort_dir, "RID_*"))):
            rid = os.path.basename(subj_dir).replace("RID_", "")
            if rid not in keep:
                continue
            pet = os.path.join(subj_dir, "PET_MNISpace_SUVR_CerebellumNorm.nii")
            if not os.path.exists(pet):
                pet = pet + ".gz"
            if os.path.exists(pet) and rid in rid_to_mri and rid in rid_to_cond:
                pet_paths.append(pet)
                mri_paths.append(rid_to_mri[rid])
                cond_vals.append(rid_to_cond[rid])
                rids.append(rid)

    print(f"Matched subjects (controls_322, {mode}): {len(pet_paths)}")

    _paint = os.environ.get("TAUGENNET_PAINT", "suvr").lower()
    if suvr_input and _paint == "atrophy":
        # PET-independent: paint atrophy, no SUVR needed -> keep the full cohort
        suvr_vals = list(cond_vals)
        print(f"[paint=atrophy] channels = MRI + atrophy; full cohort ({len(pet_paths)})")
    elif suvr_input or suvr_as_cond:
        rid_to_suvr = _build_rid_to_suvr(controls_dir)
        keep_idx = [i for i, rid in enumerate(rids) if rid in rid_to_suvr]
        dropped = len(rids) - len(keep_idx)
        if dropped:
            print(f"Dropped {dropped} subjects with no regional SUVR row")
        pet_paths = [pet_paths[i] for i in keep_idx]
        mri_paths = [mri_paths[i] for i in keep_idx]
        cond_vals = [cond_vals[i] for i in keep_idx]
        rids      = [rids[i] for i in keep_idx]
        if suvr_input:
            if _paint == "both":
                suvr_vals = [np.stack([cond_vals[i], rid_to_suvr[rids[i]]])
                             for i in range(len(rids))]          # (2,86) per subject
                print(f"[paint=both] channels = MRI + atrophy + SUVR ({len(rids)} subjects)")
            else:
                suvr_vals = [rid_to_suvr[rid] for rid in rids]
        else:
            suvr_vals = [None] * len(rids)
            cond_vals = [rid_to_suvr[rid] for rid in rids]  # FiLM: SUVR replaces atrophy as cond
    else:
        suvr_vals = [None] * len(pet_paths)

    def _make_ds(idx):
        return TauPETDataset([pet_paths[i] for i in idx],
                             [mri_paths[i] for i in idx],
                             [cond_vals[i] for i in idx],
                             use_dk_mask=use_dk_mask,
                             suvr_values=[suvr_vals[i] for i in idx] if suvr_input else None)

    denom = train_frac + val_frac
    val_frac_of_dev = (val_frac / denom) if denom else 0.2
    train_idx, val_idx, test_idx = assign_indices(
        rids, fold_idx=fold_idx, n_folds=n_folds, seed=seed,
        val_frac_of_dev=val_frac_of_dev, test_dict=test_dict, dev_dict=dev_dict,
    )
    train_ds, val_ds, test_ds = _make_ds(train_idx), _make_ds(val_idx), _make_ds(test_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"[controls_322] {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_dataloaders(mode: str, base_dir=BASE_DIR, batch_size=BATCH_SIZE, seed=SEED,
                      use_dk_mask=True, train_frac=0.64, val_frac=0.16,
                      fold_idx=None, n_folds=5, use_mentor_split=True, suvr_input=False,
                      suvr_as_cond=False, use_phase6_manifest=False, phase6_fold=0,
                      use_controls_322=False, controls_dir=None, mentor_split_dir=None):
    """
    mode      : 'ptau217' or 'atrophy'
    train_frac: fraction of subjects for training (default 0.64; 80% of 80%)
    val_frac  : fraction of subjects for validation (default 0.16; 20% of 80%); test absorbs remainder
    fold_idx  : if set (0-indexed), use k-fold CV splitting — test set is fixed last 20%;
                folds rotate over the 80% train+val pool. Overrides train_frac/val_frac.
    n_folds   : number of CV folds (default 5)
    use_mentor_split: if True (default), use Anil's fixed split from
                data/raw/{heldout_test_split.csv,train_val_split.csv}: the 47
                heldout_test subjects are the test set, and the 5 CV folds are
                stratified over the 188 dev subjects (subjects in neither file are
                dropped). If False, fall back to the legacy random fraction/fold split.
    Returns: train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    """
    if use_controls_322:
        if controls_dir is None:
            controls_dir = os.path.join(base_dir, "controls_322")
        return _build_controls_322_dataloaders(
            mode, base_dir, controls_dir, batch_size, seed, use_dk_mask,
            train_frac, val_frac, fold_idx, n_folds, suvr_input, suvr_as_cond)

    if use_phase6_manifest:
        manifest_path = os.path.join(
            os.environ.get("TAUGENNET_ROOT", "/scratch/network/sz3962/taugennet"),
            "results", "records", "phase6_manifest_v4.csv")
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["fold"] == phase6_fold]
        train_rows = manifest[manifest["split"].isin(["train", "val"])]
        test_rows = manifest[manifest["split"] == "test"]
        print(f"[phase6] cohort: {manifest['group'].value_counts().to_dict()}  "
              f"(fold {phase6_fold}, single final split — train+val merged, no held-out val)")

        def _phase6_ds(rows):
            return TauPETDataset(
                list(rows["pet"]), list(rows["mri"]),
                [np.zeros(1, dtype=np.float32)] * len(rows),  # cond unused: diagnosis not a model input
                use_dk_mask=use_dk_mask)

        train_ds = _phase6_ds(train_rows)
        val_ds = _phase6_ds(test_rows.iloc[0:0])   # empty — no held-out val, matches no-early-stopping path
        test_ds = _phase6_ds(test_rows)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)
        print(f"[phase6 fold {phase6_fold}] {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
        return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

    assert not (suvr_input and suvr_as_cond), "suvr_input and suvr_as_cond are mutually exclusive"
    if mode == "ptau217_mlp":
        mode = "ptau217"  # MLP conditioner reuses ptau217 data
    if mode not in ("ptau217", "atrophy"):
        raise ValueError(f"mode must be 'ptau217' or 'atrophy', got {mode!r}")

    rid_to_mri = _build_rid_to_mri(base_dir)
    print(f"MRI subjects found: {len(rid_to_mri)}")

    if mode == "atrophy":
        rid_to_cond = _build_rid_to_atrophy(base_dir)
        print(f"Atrophy vectors loaded: {len(rid_to_cond)}")
    else:
        rid_to_cond = _build_rid_to_ptau(ADNI_FLUID_CSV)
        print(f"p-tau217 values loaded: {len(rid_to_cond)}")

    # Mentor's fixed split (Anil): test = heldout_test_split.csv, CV pool =
    # train_val_split.csv. Subjects in neither file are dropped (`keep`).
    if use_mentor_split:
        test_dict, dev_dict = load_mentor_splits(mentor_split_dir or base_dir)
        keep = keep_set(test_dict, dev_dict)
        print(f"Mentor split: {len(test_dict)} heldout_test + {len(dev_dict)} dev "
              f"(local-only subjects not in either file are dropped)"
              + (f"  [split source: {mentor_split_dir}]" if mentor_split_dir else ""))
    else:
        test_dict = dev_dict = keep = None

    pet_paths, mri_paths, cond_vals, rids = [], [], [], []
    for cohort in ["AD", "MCI"]:
        cohort_dir = os.path.join(base_dir, "cerebellumNormalized_AD_MCI", cohort)
        if not os.path.exists(cohort_dir):
            print(f"Warning: {cohort_dir} not found, skipping")
            continue
        for subj_dir in sorted(glob.glob(os.path.join(cohort_dir, "RID_*"))):
            rid = os.path.basename(subj_dir).replace("RID_", "")
            if keep is not None and rid not in keep:
                continue  # not part of the mentor's exact split → drop
            pet = os.path.join(subj_dir, "PET_MNISpace_SUVR_CerebellumNorm.nii")
            if not os.path.exists(pet):
                pet = pet + ".gz"
            if os.path.exists(pet) and rid in rid_to_mri and rid in rid_to_cond:
                pet_paths.append(pet)
                mri_paths.append(rid_to_mri[rid])
                cond_vals.append(rid_to_cond[rid])
                rids.append(rid)

    print(f"Matched subjects ({mode}): {len(pet_paths)}")

    if suvr_input:
        rid_to_suvr = _build_rid_to_suvr(base_dir)
        keep_idx = [i for i, rid in enumerate(rids) if rid in rid_to_suvr]
        dropped = len(rids) - len(keep_idx)
        if dropped:
            print(f"Dropped {dropped} subjects with no regional SUVR row")
        pet_paths = [pet_paths[i] for i in keep_idx]
        mri_paths = [mri_paths[i] for i in keep_idx]
        cond_vals = [cond_vals[i] for i in keep_idx]
        rids      = [rids[i] for i in keep_idx]
        suvr_vals = [rid_to_suvr[rid] for rid in rids]
    else:
        suvr_vals = [None] * len(pet_paths)

    if suvr_as_cond:
        rid_to_suvr = _build_rid_to_suvr(base_dir)
        keep_idx = [i for i, rid in enumerate(rids) if rid in rid_to_suvr]
        dropped = len(rids) - len(keep_idx)
        if dropped:
            print(f"Dropped {dropped} subjects with no regional SUVR row (FiLM cond)")
        pet_paths = [pet_paths[i] for i in keep_idx]
        mri_paths = [mri_paths[i] for i in keep_idx]
        rids      = [rids[i] for i in keep_idx]
        cond_vals = [rid_to_suvr[rid] for rid in rids]  # SUVR replaces atrophy/ptau as the cond vector

    n = len(pet_paths)

    def _make_ds(idx):
        return TauPETDataset([pet_paths[i] for i in idx],
                             [mri_paths[i] for i in idx],
                             [cond_vals[i] for i in idx],
                             use_dk_mask=use_dk_mask,
                             suvr_values=[suvr_vals[i] for i in idx] if suvr_input else None)

    if use_mentor_split:
        # Fixed test = the 47 heldout RIDs; stratified 5-fold CV over the 188 dev RIDs.
        # rids is aligned (unshuffled) with the lists above, so positional indices map directly.
        denom = train_frac + val_frac
        val_frac_of_dev = (val_frac / denom) if denom else 0.2
        train_idx, val_idx, test_idx = assign_indices(
            rids, fold_idx=fold_idx, n_folds=n_folds, seed=seed,
            val_frac_of_dev=val_frac_of_dev, test_dict=test_dict, dev_dict=dev_dict,
        )
        train_ds, val_ds, test_ds = _make_ds(train_idx), _make_ds(val_idx), _make_ds(test_idx)
    else:
        # Legacy random split (kept for reproducing pre-split experiments).
        # Shuffle before splitting so AD and MCI subjects are mixed in all sets
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        pet_paths = [pet_paths[i] for i in indices]
        mri_paths = [mri_paths[i] for i in indices]
        cond_vals = [cond_vals[i] for i in indices]
        if fold_idx is not None:
            trainval_n = int(0.80 * n)
            fold_size  = trainval_n // n_folds
            val_start  = fold_idx * fold_size
            val_end    = (val_start + fold_size) if fold_idx < n_folds - 1 else trainval_n
            train_idx  = list(range(0, val_start)) + list(range(val_end, trainval_n))
            val_idx    = list(range(val_start, val_end))
            train_ds = _make_ds(train_idx)
            val_ds   = _make_ds(val_idx)
            test_ds  = _make_ds(list(range(trainval_n, n)))
            print(f"[legacy] CV fold {fold_idx}/{n_folds}: "
                  f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
        else:
            train_n = int(train_frac * n)
            val_n   = int(val_frac * n)
            train_ds = _make_ds(list(range(train_n)))
            val_ds   = _make_ds(list(range(train_n, train_n + val_n)))
            test_ds  = _make_ds(list(range(train_n + val_n, n)))
            test_frac = round(1.0 - train_frac - val_frac, 2)
            print(f"[legacy] Split ({int(train_frac*100)}/{int(val_frac*100)}/{int(test_frac*100)}): "
                  f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for mode in ("ptau217", "atrophy"):
        print(f"\n=== mode={mode} ===")
        train_ds, val_ds, test_ds, _, _, _ = build_dataloaders(mode)
        pet, mri, cond = train_ds[0]
        print(f"  PET  : {pet.shape}  range [{pet.min():.2f}, {pet.max():.2f}]")
        print(f"  MRI  : {mri.shape}  range [{mri.min():.2f}, {mri.max():.2f}]")
        print(f"  cond : {cond.shape}  sample {cond[:3].tolist()}")
        orig_min, orig_max = train_ds.get_pet_norms(0)
        print(f"  PET SUVR range (orig): [{orig_min:.3f}, {orig_max:.3f}]")
