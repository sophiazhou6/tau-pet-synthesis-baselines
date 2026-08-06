#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
dataset_spatial.py – ADNI dataset loader for spatial atrophy conditioning.

Returns (pet, mri, atrophy_vec, atrophy_map, diagnosis) per sample:
  pet, mri      – (1, H, W, D) float32, normalised to [0, 1]
  atrophy_vec   – (86,) float32 regional atrophy z-scores
  atrophy_map   – (1, H, W, D) float32 spatial map; each voxel = atrophy z-score
                  of its DK86 region; out-of-atlas voxels = 0.0
  diagnosis     – int tensor  0=MCI, 1=AD  (from subject folder path)

When ``use_tissue=True`` an extra ``tissue_onehot`` is inserted before
``diagnosis``: (pet, mri, atrophy_vec, atrophy_map, tissue_onehot, diagnosis)
  tissue_onehot – (3, H, W, D) float32 one-hot of T1_seg_in_MNI.nii.gz
                  labels 1=GM, 2=WM, 3=CSF; masked to valid voxels.

Masking: mentor-provided approach — valid voxels are where mri > 0 AND tau > 0,
mapped through affines to check inside the MNI DK atlas.

Requires: data/raw/mni_dk_atlas.nii.gz  (desikanKilliany86MNI.nii.gz)
          data/raw/regional_atrophy_zscores.csv
"""

import os
import glob
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from .config import BASE_DIR, VOL_SHAPE, BATCH_SIZE, SEED, ADNI_FLUID_CSV
from .atlas_labels import load_atlas, compute_valid_mask, build_atrophy_volume, get_default_atlas_path
from .splits import load_mentor_splits, assign_indices, keep_set

# ── Region columns (same ordering as dataset.py) ──────────────────────────────
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

# Regional tau-SUVR conditioning (mentor 2026-07). The SUVR CSV
# (regional_SUVR_cerebellumNormalizedGTScan1.csv) columns are the bare _ATLAS[i] names
# (i=1..86) in atlas-label order — no suffix — so they read straight into the SAME
# painter as the atrophy map. Enabled by env var TAUGENNET_SPATIAL_MAP=suvr, which paints
# regional SUVR into the spatial-conditioning channel in place of atrophy z-scores.
SUVR_COLS = [_ATLAS[i] for i in range(1, 87)]


# ── Dataset class ─────────────────────────────────────────────────────────────

class SpatialTauPETDataset(Dataset):
    """
    Returns (pet, mri, atrophy_vec, atrophy_map, diagnosis) where:
      pet, mri      – (1, H, W, D) float32, normalised to [0, 1]
      atrophy_vec   – (86,) float32 regional atrophy z-scores
      atrophy_map   – (1, H, W, D) float32 spatial atrophy map
      diagnosis     – int tensor, 0=MCI / 1=AD
    """

    # T1 segmentation labels (FSL FAST convention): 1=GM, 2=WM, 3=CSF
    TISSUE_LABELS = (1, 2, 3)

    def __init__(self, pet_paths, mri_paths, atrophy_vecs, diagnoses,
                 atlas_data, inv_atlas_affine, use_dk_mask=True, use_tissue=False,
                 cond_mode="atrophy", ptau_vals=None, suvr_vecs=None):
        self.suvr_vecs = suvr_vecs   # not None -> paint a 2nd channel (SUVR)
        self.pet_paths       = pet_paths
        self.mri_paths       = mri_paths
        self.atrophy_vecs    = atrophy_vecs      # list of (86,) np.float32 arrays (always: builds the map)
        self.diagnoses       = diagnoses          # list of int (0 or 1)
        self._atlas_data     = atlas_data         # (aH, aW, aD) int32
        self._inv_atlas_aff  = inv_atlas_affine   # (4,4)
        self.use_dk_mask     = use_dk_mask
        self.use_tissue      = use_tissue
        # Shared global DK86 binary mask at VOL_SHAPE — IDENTICAL to dataset_final's
        # mask so PET/MRI land in the exact support (and per-subject normalization)
        # the AE was trained on. Using the broader per-subject mri>0∩atlas mask here
        # was out-of-distribution for the AE and collapsed reconstruction (0.98→0.52).
        if use_dk_mask:
            binary = (atlas_data > 0).astype(np.float32)                 # labels 1-86 → 1.0
            mask_t = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)  # (1,1,aH,aW,aD)
            self._dk_mask = F.interpolate(mask_t, size=VOL_SHAPE,
                                          mode="nearest").squeeze(0)     # (1, *VOL_SHAPE)
        else:
            self._dk_mask = None
        # cond_mode selects the cross-attention input returned in slot 3 ("cond_vec"):
        #   "atrophy"/"none" -> atrophy_vec (86,);  "ptau217" -> ptau (1,).
        # The spatial atrophy MAP is always built from atrophy_vecs regardless.
        self.cond_mode       = cond_mode
        self.ptau_vals       = ptau_vals          # list of (1,) np.float32, only for ptau217

    @staticmethod
    def _seg_path(mri_path):
        """Derive the tissue-segmentation path from the MRI path (see evaluate_tissue.py)."""
        return mri_path.replace("T1_to_MNI_nonlin.nii.gz", "T1_seg_in_MNI.nii.gz")

    def _load_volume(self, path):
        img = nib.load(path)
        vol = img.get_fdata().astype(np.float32)
        return vol, img.affine

    def _to_tensor(self, vol: np.ndarray, mask: torch.Tensor = None) -> torch.Tensor:
        t = torch.from_numpy(vol).unsqueeze(0)           # (1, H, W, D)
        t = F.interpolate(t.unsqueeze(0), size=VOL_SHAPE,
                          mode="trilinear", align_corners=False).squeeze(0)
        if mask is not None:
            t = t * mask                                 # (1, *VOL_SHAPE) — already at target res
        vmin, vmax = t.min(), t.max()
        return (t - vmin) / (vmax - vmin + 1e-8)

    def _load_tissue(self, mri_path, mask: np.ndarray = None) -> torch.Tensor:
        """Load T1 seg → one-hot (3, *VOL_SHAPE) for labels 1=GM, 2=WM, 3=CSF.

        Categorical: nearest-neighbour resampling only (never trilinear). The
        valid-voxel mask is applied at VOL_SHAPE, mirroring ``_to_tensor``.
        """
        seg_vol, _ = self._load_volume(self._seg_path(mri_path))
        seg = torch.from_numpy(seg_vol)                                  # (H0, W0, D0)
        # one-hot at native resolution, then nearest-resample each channel
        onehot = torch.stack([(seg == lbl).float() for lbl in self.TISSUE_LABELS])  # (3,H0,W0,D0)
        onehot = F.interpolate(onehot.unsqueeze(0), size=VOL_SHAPE,
                               mode="nearest").squeeze(0)                # (3, *VOL_SHAPE)
        if mask is not None:
            onehot = onehot * mask                                      # (1,*VOL_SHAPE) broadcasts over 3 ch
        return onehot

    def __len__(self):
        return len(self.pet_paths)

    def get_pet_norms(self, idx):
        """(min, max) used to normalize the PET at idx — over the SAME masked support as
        _to_tensor. Used to unnormalize to true SUVR; without it, eval falls back to the
        RAW NiFTI max (out-of-brain hot voxels ~20) and inflates SUVR ~4x, corrupting
        SET1/SET3."""
        vol, _ = self._load_volume(self.pet_paths[idx])
        t = torch.from_numpy(vol).unsqueeze(0)
        t = F.interpolate(t.unsqueeze(0), size=VOL_SHAPE,
                          mode="trilinear", align_corners=False).squeeze(0)
        if self.use_dk_mask and self._dk_mask is not None:
            t = t * self._dk_mask
        return float(t.min()), float(t.max())

    def __getitem__(self, idx):
        mri_vol, mri_affine = self._load_volume(self.mri_paths[idx])
        pet_vol, _          = self._load_volume(self.pet_paths[idx])

        # Shared global DK86 binary mask (already at VOL_SHAPE) — the exact support and
        # per-subject normalization the AE was trained on (matches dataset_final). The
        # atrophy MAP built below is still zero outside the atlas by construction.
        mask = self._dk_mask if self.use_dk_mask else None

        pet         = self._to_tensor(pet_vol, mask)
        mri         = self._to_tensor(mri_vol, mask)
        # cross-attention input ("cond_vec"): ptau (1,) in ptau217 mode, else atrophy z-scores (86,)
        if self.cond_mode in ("ptau217", "ptau217_mlp"):
            cond_vec = torch.tensor(self.ptau_vals[idx], dtype=torch.float32)
        else:
            cond_vec = torch.tensor(self.atrophy_vecs[idx], dtype=torch.float32)
        # spatial atrophy MAP is always built from the full 86-region atrophy vector
        atrophy_map = build_atrophy_volume(
            mri_vol.shape, mri_affine,
            self._atlas_data, self._inv_atlas_aff,
            self.atrophy_vecs[idx]
        )
        # Resize atrophy_map to VOL_SHAPE
        atrophy_map = F.interpolate(atrophy_map.unsqueeze(0), size=VOL_SHAPE,
                                    mode="trilinear", align_corners=False).squeeze(0)
        if getattr(self, "suvr_vecs", None) is not None:
            _suvr_map = build_atrophy_volume(
                mri_vol.shape, mri_affine,
                self._atlas_data, self._inv_atlas_aff,
                self.suvr_vecs[idx])
            _suvr_map = F.interpolate(_suvr_map.unsqueeze(0), size=VOL_SHAPE,
                                      mode="trilinear", align_corners=False).squeeze(0)
            atrophy_map = torch.cat([atrophy_map, _suvr_map], dim=0)   # (2, *VOL_SHAPE)
        diagnosis   = torch.tensor(self.diagnoses[idx], dtype=torch.long)
        if self.use_tissue:
            tissue = self._load_tissue(self.mri_paths[idx], mask)
            return pet, mri, cond_vec, atrophy_map, tissue, diagnosis
        return pet, mri, cond_vec, atrophy_map, diagnosis


# ── Loaders ───────────────────────────────────────────────────────────────────

def _build_rid_to_mri_and_diagnosis(base_dir):
    rid_to_mri  = {}
    rid_to_diag = {}
    for cohort, label in [("1mm_parcellated_AD_subj", 1), ("1mm_parcellated_MCI_subj", 0), ("1mm_parcellated_CN_subj", 2)]:
        for subj_dir in sorted(glob.glob(os.path.join(base_dir, cohort, "*"))):
            rid = os.path.basename(subj_dir).split("_")[-1]
            mri = os.path.join(subj_dir, "T1_to_MNI_nonlin.nii.gz")
            if os.path.exists(mri):
                rid_to_mri[rid]  = mri
                rid_to_diag[rid] = label
    return rid_to_mri, rid_to_diag


def _build_rid_to_atrophy(base_dir):
    df      = pd.read_csv(os.path.join(base_dir, "regional_atrophy_zscores.csv"))
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
        rid_to_atrophy[rid] = np.nan_to_num(vec, nan=0.0)
    if duplicates:
        print(f"Note: {duplicates} duplicate RIDs in atrophy CSV — keeping latest")
    if skipped:
        print(f"Skipped {skipped} all-NaN atrophy rows")
    return rid_to_atrophy


def _build_rid_to_suvr(base_dir):
    """Map RID -> (86,) regional tau-SUVR vector in atlas-label order.

    The SUVR CSV columns are the bare _ATLAS[i] names (i=1..86), already in atlas-label
    order, so SUVR_COLS reads straight into build_atrophy_volume's painter — identical
    mechanism to the atrophy map, only a different source value per region.
    """
    df      = pd.read_csv(os.path.join(base_dir, "regional_SUVR_cerebellumNormalizedGTScan1.csv"))
    missing = [c for c in SUVR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"SUVR CSV missing columns: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    rid_to_suvr = {}
    skipped = duplicates = 0
    for _, row in df.iterrows():
        rid = str(int(row["RID"]))
        vec = row[SUVR_COLS].values.astype(np.float32)
        if np.all(np.isnan(vec)):
            skipped += 1
            continue
        if rid in rid_to_suvr:
            duplicates += 1
        rid_to_suvr[rid] = np.nan_to_num(vec, nan=0.0)
    if duplicates:
        print(f"Note: {duplicates} duplicate RIDs in SUVR CSV — keeping latest")
    if skipped:
        print(f"Skipped {skipped} all-NaN SUVR rows")
    return rid_to_suvr


def _build_rid_to_ptau(fluid_csv):
    """Map RID -> (1,) p-tau217 plasma value (column pT217_F). Reused from dataset_combined."""
    df = pd.read_csv(fluid_csv)
    return {
        str(int(row["RID"])): np.array([row["pT217_F"]], dtype=np.float32)
        for _, row in df.iterrows()
        if pd.notna(row["pT217_F"])
    }


def build_dataloaders(base_dir=BASE_DIR, batch_size=BATCH_SIZE, seed=SEED,
                      use_dk_mask=True, train_frac=0.64, val_frac=0.16,
                      atlas_path=None, fold_idx=None, n_folds=5, use_tissue=False,
                      cond_mode="atrophy", use_mentor_split=True, no_val_split=False,
                      use_controls_322=False, controls_dir=None):
    """
    Returns train_ds, val_ds, test_ds, train_loader, val_loader, test_loader.

    use_tissue: if True, also require a T1_seg_in_MNI.nii.gz next to each MRI and
                return a (3, *VOL_SHAPE) one-hot tissue map per sample.
    cond_mode : cross-attention input. "atrophy"/"none" -> atrophy z-scores (the
                spatial atrophy map is present regardless); "ptau217" -> require a
                plasma p-tau217 value per subject and return it as the cond vector.
    use_mentor_split: if True (default), use Anil's fixed split from
                data/raw/{heldout_test_split.csv,train_val_split.csv}: the 47
                heldout_test subjects are the test set, and the 5 CV folds are
                stratified over the 188 dev subjects (subjects in neither file are
                dropped). If False, fall back to the legacy random fraction/fold split.
    """
    atlas_path = atlas_path or get_default_atlas_path()
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(
            f"DK86 atlas not found at {atlas_path}.\n"
            "Place desikanKilliany86MNI.nii.gz at data/raw/mni_dk_atlas.nii.gz"
        )
    print(f"Loading atlas from {atlas_path} …", flush=True)
    atlas_data, _, inv_atlas_affine = load_atlas(atlas_path)
    print(f"Atlas loaded — shape {atlas_data.shape}, "
          f"regions: {(np.unique(atlas_data) > 0).sum()} labels")

    rid_to_mri, rid_to_diag = _build_rid_to_mri_and_diagnosis(base_dir)
    rid_to_atrophy           = _build_rid_to_atrophy(base_dir)
    # Spatial-map source: atrophy (default) or regional SUVR (env TAUGENNET_SPATIAL_MAP=suvr).
    # In "suvr" mode the painted spatial channel carries regional tau-SUVR instead of atrophy
    # z-scores; the cross-attention cond_vec, model, and checkpoint are all unchanged.
    _map_mode     = os.environ.get("TAUGENNET_SPATIAL_MAP", "atrophy").lower()
    use_suvr_map  = _map_mode == "suvr"
    use_both_maps = _map_mode == "both"   # ch0 = atrophy, ch1 = SUVR
    if use_controls_322 and controls_dir is None:
        controls_dir = os.path.join(base_dir, "controls_322")
    rid_to_suvr  = _build_rid_to_suvr(controls_dir if use_controls_322 else base_dir) \
                   if (use_suvr_map or use_both_maps) else {}
    use_ptau = cond_mode in ("ptau217", "ptau217_mlp")
    # Cohort-matching only: restrict to subjects WITH a p-tau value without conditioning on it,
    # so a no-plasma arm can be compared against a plasma arm on identical subjects.
    _require_ptau = os.environ.get("TAUGENNET_REQUIRE_PTAU", "0") == "1"
    rid_to_ptau = _build_rid_to_ptau(ADNI_FLUID_CSV) if (use_ptau or _require_ptau) else {}
    print(f"MRI subjects: {len(rid_to_mri)}  |  Atrophy subjects: {len(rid_to_atrophy)}"
          + (f"  |  SUVR subjects: {len(rid_to_suvr)}" if use_suvr_map else "")
          + (f"  |  p-tau217 subjects: {len(rid_to_ptau)}" if use_ptau else ""))
    if use_suvr_map:
        print("[spatial map = SUVR] painting regional tau-SUVR into the spatial channel "
              "(atrophy z-scores NOT used for the map)")

    # Mentor's fixed split (Anil): test = heldout_test_split.csv, CV pool =
    # train_val_split.csv. Subjects in neither file are dropped (`keep`).
    if use_controls_322:
        from .dataset_final import _load_controls_322_splits
        use_mentor_split = True   # controls_322 reuses the fixed-split index path below
        test_dict, dev_dict = _load_controls_322_splits(controls_dir)
        keep = keep_set(test_dict, dev_dict)
        print(f"controls_322 split: {len(test_dict)} heldout_test + {len(dev_dict)} dev "
              f"(subjects not in either file are dropped)")
    elif use_mentor_split:
        test_dict, dev_dict = load_mentor_splits(base_dir)
        keep = keep_set(test_dict, dev_dict)
        print(f"Mentor split: {len(test_dict)} heldout_test + {len(dev_dict)} dev "
              f"(local-only subjects not in either file are dropped)")
    else:
        test_dict = dev_dict = keep = None

    pet_paths, mri_paths, atrophy_vecs, diagnoses, ptau_vals, rids = [], [], [], [], [], []
    suvr_vecs = []
    for cohort in ["AD", "MCI", "CN"]:
        cohort_dir = os.path.join(base_dir, "cerebellumNormalized_AD_MCI", cohort)
        if not os.path.exists(cohort_dir):
            print(f"Warning: {cohort_dir} not found, skipping")
            continue
        for subj_dir in sorted(glob.glob(os.path.join(cohort_dir, "RID_*"))):
            rid = os.path.basename(subj_dir).replace("RID_", "")
            if keep is not None and rid not in keep:
                continue  # not part of the mentor's exact split → drop
            pet = os.path.join(subj_dir, "PET_MNISpace_SUVR_CerebellumNorm.nii.gz")
            if not os.path.exists(pet):
                pet = pet.replace(".nii.gz", ".nii")
            if os.path.exists(pet) and rid in rid_to_mri and rid in rid_to_atrophy:
                if use_tissue and not os.path.exists(
                        SpatialTauPETDataset._seg_path(rid_to_mri[rid])):
                    continue  # skip subjects without a tissue segmentation
                if (use_ptau or _require_ptau) and rid not in rid_to_ptau:
                    continue  # skip subjects without a plasma p-tau217 value
                if (use_suvr_map or use_both_maps) and rid not in rid_to_suvr:
                    continue  # SUVR-map mode: skip subjects without a regional SUVR row
                pet_paths.append(pet)
                mri_paths.append(rid_to_mri[rid])
                # SUVR-map mode paints the spatial channel from regional SUVR; the
                # atrophy_vecs slot carries that vector so the painter stays unchanged.
                atrophy_vecs.append(rid_to_suvr[rid] if use_suvr_map else rid_to_atrophy[rid])
                suvr_vecs.append(rid_to_suvr[rid] if use_both_maps else None)
                diagnoses.append(rid_to_diag[rid])
                ptau_vals.append(rid_to_ptau[rid] if use_ptau else None)
                rids.append(rid)

    tag = f"spatial, cond={cond_mode}{', +tissue' if use_tissue else ''}"
    print(f"Matched subjects ({tag}): {len(pet_paths)}")

    n = len(pet_paths)

    def _make_ds(idx_list):
        return SpatialTauPETDataset(
            [pet_paths[i]    for i in idx_list],
            [mri_paths[i]    for i in idx_list],
            [atrophy_vecs[i] for i in idx_list],
            [diagnoses[i]    for i in idx_list],
            atlas_data, inv_atlas_affine,
            use_dk_mask=use_dk_mask, use_tissue=use_tissue,
            cond_mode=cond_mode, ptau_vals=[ptau_vals[i] for i in idx_list],
            suvr_vecs=([suvr_vecs[i] for i in idx_list] if use_both_maps else None),
        )

    if use_mentor_split:
        # Fixed test = the 47 heldout RIDs; stratified 5-fold CV over the 188 dev RIDs.
        # rids is aligned (unshuffled) with the lists above, so positional indices map directly.
        denom = train_frac + val_frac
        val_frac_of_dev = (val_frac / denom) if denom else 0.2
        train_idx, val_idx, test_idx = assign_indices(
            rids, fold_idx=fold_idx, n_folds=n_folds, seed=seed,
            val_frac_of_dev=val_frac_of_dev, test_dict=test_dict, dev_dict=dev_dict,
        )
        if no_val_split:
            # Fold val back into train -> train on ALL dev; val empty.
            train_idx = train_idx + val_idx
            val_idx = []
        train_ds = _make_ds(train_idx)
        val_ds   = _make_ds(val_idx)
        test_ds  = _make_ds(test_idx)
    else:
        # Legacy random split (kept for reproducing pre-split experiments).
        rng     = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        pet_paths    = [pet_paths[i]    for i in indices]
        mri_paths    = [mri_paths[i]    for i in indices]
        atrophy_vecs = [atrophy_vecs[i] for i in indices]
        diagnoses    = [diagnoses[i]    for i in indices]
        ptau_vals    = [ptau_vals[i]    for i in indices]
        if fold_idx is not None:
            trainval_n = int(0.80 * n)
            fold_size  = trainval_n // n_folds
            val_start  = fold_idx * fold_size
            val_end    = (val_start + fold_size) if fold_idx < n_folds - 1 else trainval_n
            train_idx  = list(range(0, val_start)) + list(range(val_end, trainval_n))
            val_idx    = list(range(val_start, val_end))
            train_ds   = _make_ds(train_idx)
            val_ds     = _make_ds(val_idx)
            test_ds    = _make_ds(list(range(trainval_n, n)))
            print(f"[legacy] CV fold {fold_idx}/{n_folds}: "
                  f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
        else:
            train_n  = int(train_frac * n)
            val_n    = int(val_frac * n)
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
