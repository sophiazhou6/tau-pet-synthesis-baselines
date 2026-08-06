"""
dataset_combined.py – ADNI dataset loader for combined atrophy + ptau217 conditioning.

mode='combined'
    Conditioning data: 87-dim vector = [atrophy z-scores (86), ptau217 scalar (1)].
    Only subjects with MRI, PET, atrophy z-scores, AND ptau217 values are included.
    Split: 80 / 10 / 10  train / val / test.

Both conditioning signals are concatenated into a single (87,) tensor so the
CombinedConditioner can split them back apart (first 86 = atrophy, last 1 = ptau).

Masking: Uses Desikan-Killiany (DK) atlas (86 regions) via RegionLabelMap.py.
Only voxels within the 86 DK regions are retained; non-brain regions are masked out.
"""

import os
import glob
import random

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


# ── Dataset class ─────────────────────────────────────────────────────────────

class TauPETDataset(Dataset):
    """
    Returns (pet, mri, cond_data) where:
      pet, mri   – (1, H, W, D) float tensor normalised to [0, 1]
      cond_data  – (87,) = [atrophy z-scores (86) ‖ ptau217 scalar (1)]

    Volumes are masked to include only the 86 Desikan-Killiany atlas regions.
    """

    def __init__(self, pet_paths, mri_paths, cond_values, use_dk_mask=True):
        self.pet_paths   = pet_paths
        self.mri_paths   = mri_paths
        self.cond_values = cond_values
        self.use_dk_mask = use_dk_mask
        if use_dk_mask:
            atlas_data, _, _ = load_atlas(get_default_atlas_path())
            binary = (atlas_data > 0).astype(np.float32)
            mask_t = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)
            self._dk_mask = F.interpolate(mask_t, size=VOL_SHAPE,
                                          mode="nearest").squeeze(0)
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
        vmin, vmax = vol.min(), vol.max()
        return (vol - vmin) / (vmax - vmin + 1e-8)

    def __len__(self):
        return len(self.pet_paths)

    def __getitem__(self, idx):
        mask = self._get_dk_mask(idx) if self.use_dk_mask else None
        pet  = self._load(self.pet_paths[idx], mask=mask)
        mri  = self._load(self.mri_paths[idx], mask=mask)
        cond = torch.tensor(self.cond_values[idx], dtype=torch.float32)
        return pet, mri, cond

    def get_pet_norms(self, idx):
        """(min, max) used to normalize the PET at idx — computed over the SAME masked
        support as _load. Callers use it to unnormalize back to true SUVR. Without this,
        evaluate_final/glass_brain fall back to the RAW NiFTI max (which includes
        out-of-brain hot voxels ~20) and inflate SUVR ~4x — corrupting SET1/SET3."""
        import nibabel as nib
        mask = self._get_dk_mask(idx) if self.use_dk_mask else None
        vol = nib.load(self.pet_paths[idx]).get_fdata().astype(np.float32)
        vol = F.interpolate(torch.tensor(vol).unsqueeze(0).unsqueeze(0),
                            size=VOL_SHAPE, mode="trilinear", align_corners=False).squeeze(0)
        if mask is not None:
            vol = vol * mask
        return float(vol.min()), float(vol.max())


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
    skipped = duplicates = 0
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


def _build_controls_322_dataloaders_combined(base_dir, controls_dir, batch_size, seed,
                                              use_dk_mask, train_frac, val_frac, fold_idx, n_folds):
    """controls_322-aware loader for combined (atrophy+ptau217) mode -- mirrors
    dataset_final.py's _build_controls_322_dataloaders, extended with ptau217 lookup
    (the atrophy-only version there has no ptau217 support at all)."""
    from . import dataset_final as _df
    rid_to_mri = _df._build_rid_to_mri(base_dir, extra_cohorts=["1mm_parcellated_CN_subj"])
    print(f"MRI subjects found (incl. CN): {len(rid_to_mri)}")
    rid_to_atrophy = _build_rid_to_atrophy(base_dir)
    print(f"Atrophy vectors loaded: {len(rid_to_atrophy)}")
    rid_to_ptau = _build_rid_to_ptau(ADNI_FLUID_CSV)
    print(f"p-tau217 values loaded: {len(rid_to_ptau)}")

    test_dict, dev_dict = _df._load_controls_322_splits(controls_dir)
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
            if (os.path.exists(pet) and rid in rid_to_mri
                    and rid in rid_to_atrophy and rid in rid_to_ptau):
                pet_paths.append(pet)
                mri_paths.append(rid_to_mri[rid])
                cond_vals.append(np.concatenate([rid_to_atrophy[rid], rid_to_ptau[rid]]))
                rids.append(rid)

    print(f"Matched subjects (controls_322, combined): {len(pet_paths)}")

    def _make_ds(idx):
        return TauPETDataset([pet_paths[i] for i in idx],
                             [mri_paths[i] for i in idx],
                             [cond_vals[i] for i in idx], use_dk_mask=use_dk_mask)

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


def build_dataloaders(mode: str = "combined", base_dir=BASE_DIR,
                      batch_size=BATCH_SIZE, seed=SEED, use_dk_mask=True,
                      train_frac=0.64, val_frac=0.16,
                      fold_idx=None, n_folds=5, use_mentor_split=True,
                      use_controls_322=False, controls_dir=None):
    """
    mode      : must be 'combined'
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
    cond      : (87,) = [atrophy z-scores (86) ‖ ptau217 (1)]
    Returns: train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    """
    if mode == "combined_mlp":
        mode = "combined"  # MLP conditioner reuses combined data

    if use_controls_322:
        if controls_dir is None:
            controls_dir = os.path.join(base_dir, "controls_322")
        return _build_controls_322_dataloaders_combined(
            base_dir, controls_dir, batch_size, seed, use_dk_mask,
            train_frac, val_frac, fold_idx, n_folds)
    if mode != "combined":
        raise ValueError(f"dataset_combined only supports mode='combined', got {mode!r}")

    rid_to_mri     = _build_rid_to_mri(base_dir)
    rid_to_atrophy = _build_rid_to_atrophy(base_dir)
    rid_to_ptau    = _build_rid_to_ptau(ADNI_FLUID_CSV)
    print(f"MRI subjects found: {len(rid_to_mri)}")
    print(f"Atrophy vectors loaded: {len(rid_to_atrophy)}")
    print(f"p-tau217 values loaded: {len(rid_to_ptau)}")

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
            if (os.path.exists(pet) and rid in rid_to_mri
                    and rid in rid_to_atrophy and rid in rid_to_ptau):
                pet_paths.append(pet)
                mri_paths.append(rid_to_mri[rid])
                # concatenate: (86,) atrophy + (1,) ptau → (87,)
                cond_vals.append(
                    np.concatenate([rid_to_atrophy[rid], rid_to_ptau[rid]])
                )
                rids.append(rid)

    print(f"Matched subjects (combined): {len(pet_paths)}")

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
