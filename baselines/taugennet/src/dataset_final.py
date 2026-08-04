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

    def __init__(self, pet_paths, mri_paths, cond_values, use_dk_mask=True):
        self.pet_paths   = pet_paths
        self.mri_paths   = mri_paths
        self.cond_values = cond_values  # list of np.ndarray or float
        self.use_dk_mask = use_dk_mask
        self._pet_norms  = {}           # idx -> (orig_min, orig_max), populated lazily
        # Precompute a single global DK86 atlas mask at VOL_SHAPE resolution.
        # All subjects are in MNI space, so one shared mask applies to all.
        if use_dk_mask:
            atlas_data, _, _ = load_atlas(get_default_atlas_path())
            binary = (atlas_data > 0).astype(np.float32)  # labels 1-86 → 1.0
            mask_t = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)  # (1,1,aH,aW,aD)
            self._dk_mask = F.interpolate(mask_t, size=VOL_SHAPE,
                                          mode="nearest").squeeze(0)      # (1,H,W,D)
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
        return pet, mri, cond


# ── Loaders ───────────────────────────────────────────────────────────────────

def _build_rid_to_mri(base_dir):
    rid_to_mri = {}
    for cohort in ["1mm_parcellated_AD_subj", "1mm_parcellated_MCI_subj"]:
        for subj_dir in sorted(glob.glob(os.path.join(base_dir, cohort, "*"))):
            rid = os.path.basename(subj_dir).split("_")[-1]
            mri = os.path.join(subj_dir, "T1_to_MNI_nonlin.nii.gz")
            if os.path.exists(mri):
                rid_to_mri[rid] = mri
    return rid_to_mri


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


def build_dataloaders(mode: str, base_dir=BASE_DIR, batch_size=BATCH_SIZE, seed=SEED,
                      use_dk_mask=True, train_frac=0.64, val_frac=0.16,
                      fold_idx=None, n_folds=5, use_mentor_split=True):
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
        test_dict, dev_dict = load_mentor_splits(base_dir)
        keep = keep_set(test_dict, dev_dict)
        print(f"Mentor split: {len(test_dict)} heldout_test + {len(dev_dict)} dev "
              f"(local-only subjects not in either file are dropped)")
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

    n = len(pet_paths)

    def _make_ds(idx):
        return TauPETDataset([pet_paths[i] for i in idx],
                             [mri_paths[i] for i in idx],
                             [cond_vals[i] for i in idx], use_dk_mask=use_dk_mask)

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
