"""
dataset_combined_relu.py – ADNI dataset loader for combined atrophy + ptau217 conditioning with ReLU config.

mode='combined'
    Conditioning data: 87-dim vector = [atrophy z-scores (86), ptau217 scalar (1)].
    Only subjects with MRI, PET, atrophy z-scores, AND ptau217 values are included.
    Split: 70 / 10 / 20  train / val / test.
    Uses config_relu with VOL_SHAPE = (96, 112, 96)

Both conditioning signals are concatenated into a single (87,) tensor so the
CombinedConditioner can split them back apart (first 86 = atrophy, last 1 = ptau).

Masking: Uses Desikan-Killiany (DK) atlas (86 regions).
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

from .config_relu import BASE_DIR, VOL_SHAPE, BATCH_SIZE, ADNI_FLUID_CSV, SEED

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
        self._mask_cache = {}

    def _get_dk_mask(self, idx):
        """
        Load DK atlas mask (labels 1-86 from T1_seg_in_MNI.nii.gz), cached after first load.
        Masks out non-brain regions (skull, dura, ventricles, etc.).
        """
        if idx not in self._mask_cache:
            import nibabel as nib
            seg_path = os.path.join(os.path.dirname(self.mri_paths[idx]),
                                    "T1_seg_in_MNI.nii.gz")
            seg = nib.load(seg_path).get_fdata().astype(np.uint8)
            mask = torch.from_numpy(((seg >= 1) & (seg <= 86)).astype(np.float32)).unsqueeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=VOL_SHAPE,
                                 mode="nearest").squeeze(0)
            self._mask_cache[idx] = mask
        return self._mask_cache[idx]

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


def build_dataloaders(mode: str = "combined", base_dir=BASE_DIR,
                      batch_size=BATCH_SIZE, seed=SEED, use_dk_mask=True):
    """
    mode  : must be 'combined'
    Split : 70% train / 10% val / 20% test.
    cond  : (87,) = [atrophy z-scores (86) ‖ ptau217 (1)]
    use_dk_mask : if True, mask volumes to 86 DK atlas regions; if False, no masking.
    Returns: train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    """
    if mode != "combined":
        raise ValueError(f"dataset_combined only supports mode='combined', got {mode!r}")

    rid_to_mri     = _build_rid_to_mri(base_dir)
    rid_to_atrophy = _build_rid_to_atrophy(base_dir)
    rid_to_ptau    = _build_rid_to_ptau(ADNI_FLUID_CSV)
    print(f"MRI subjects found: {len(rid_to_mri)}")
    print(f"Atrophy vectors loaded: {len(rid_to_atrophy)}")
    print(f"p-tau217 values loaded: {len(rid_to_ptau)}")

    pet_paths, mri_paths, cond_vals = [], [], []
    for cohort in ["AD", "MCI"]:
        cohort_dir = os.path.join(base_dir, "cerebellumNormalized_AD_MCI", cohort)
        if not os.path.exists(cohort_dir):
            print(f"Warning: {cohort_dir} not found, skipping")
            continue
        for subj_dir in sorted(glob.glob(os.path.join(cohort_dir, "RID_*"))):
            rid = os.path.basename(subj_dir).replace("RID_", "")
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

    print(f"Matched subjects (combined): {len(pet_paths)}")

    rng = random.Random(seed)
    indices = list(range(len(pet_paths)))
    rng.shuffle(indices)
    pet_paths = [pet_paths[i] for i in indices]
    mri_paths = [mri_paths[i] for i in indices]
    cond_vals = [cond_vals[i] for i in indices]

    n       = len(pet_paths)
    train_n = int(0.64 * n)
    val_n   = int(0.16 * n)

    train_ds = TauPETDataset(pet_paths[:train_n],              mri_paths[:train_n],              cond_vals[:train_n],              use_dk_mask=use_dk_mask)
    val_ds   = TauPETDataset(pet_paths[train_n:train_n+val_n], mri_paths[train_n:train_n+val_n], cond_vals[train_n:train_n+val_n], use_dk_mask=use_dk_mask)
    test_ds  = TauPETDataset(pet_paths[train_n+val_n:],        mri_paths[train_n+val_n:],        cond_vals[train_n+val_n:],        use_dk_mask=use_dk_mask)
    print(f"Split (64/16/20): {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
