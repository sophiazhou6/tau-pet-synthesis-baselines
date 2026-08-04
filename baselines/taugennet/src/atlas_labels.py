#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
atlas_labels.py – DK86 atlas label → REGION_COLS index mapping.

Atlas: data/raw/mni_dk_atlas.nii.gz  (desikanKilliany86MNI.nii.gz from mentor)
  - Labels 1–86 map directly to REGION_COLS indices 0–85 (label - 1)
  - Shape (97, 115, 97), MNI space
  - Note: atlas positions 77/86 are Hypothalamus L/R; TauGenNet REGION_COLS
    positions 76/85 (0-based) are LEFT/RIGHT_VENTRALDC. These two structures
    are adjacent — the mismatch is acknowledged and left as-is since the
    atrophy CSV uses VentralDC column names from ADNI.

Masking: mentor-provided _compute_valid_voxels logic — valid voxels are those
where mri > 0 AND tau > 0, mapped through affines to check inside the MNI atlas.
"""

import os
import numpy as np
import nibabel as nib
import torch

from .config import BASE_DIR, VOL_SHAPE


def get_default_atlas_path() -> str:
    return os.path.join(BASE_DIR, "mni_dk_atlas.nii.gz")


def load_atlas(atlas_path: str) -> tuple:
    """Load the DK86 MNI atlas.

    Returns
    -------
    atlas_data : (H, W, D) int32 ndarray — raw label values 0–86
    affine     : (4, 4) ndarray — atlas voxel→world affine
    inv_affine : (4, 4) ndarray — world→atlas voxel affine
    """
    img        = nib.load(atlas_path)
    atlas_data = img.get_fdata().astype(np.int32)
    affine     = img.affine
    inv_affine = np.linalg.inv(affine)
    return atlas_data, affine, inv_affine


def compute_valid_mask(mri: np.ndarray, tau: np.ndarray,
                       subject_affine: np.ndarray,
                       atlas_data: np.ndarray,
                       inv_atlas_affine: np.ndarray) -> np.ndarray:
    """Return a boolean mask of valid in-brain voxels, computed in MRI space.

    Mirrors the mentor's _compute_valid_voxels. MRI (2mm, 91×109×91) and PET
    (1mm, 182×218×182) are in the same MNI space but different resolutions, so
    the mask is built from mri > 0 alone and validated against the atlas affine.
    The mask is returned in MRI space; callers resample it to target resolution.

    Parameters
    ----------
    mri              : (H, W, D) MRI array (used for base mask)
    tau              : unused — kept for API compatibility
    subject_affine   : (4,4) affine of the MRI volume
    atlas_data       : (aH, aW, aD) int32 — output of load_atlas()[0]
    inv_atlas_affine : (4,4) — output of load_atlas()[2]

    Returns
    -------
    (H, W, D) bool mask in MRI space
    """
    mni_mask = atlas_data > 0
    base = np.isfinite(mri) & (mri > 0)
    ijk  = np.argwhere(base)
    if ijk.size == 0:
        return base

    # Map subject voxel coords → world → atlas voxel coords
    xyz      = nib.affines.apply_affine(subject_affine, ijk)
    mask_ijk = np.round(nib.affines.apply_affine(inv_atlas_affine, xyz)).astype(int)

    valid = ((mask_ijk[:, 0] >= 0) & (mask_ijk[:, 0] < mni_mask.shape[0]) &
             (mask_ijk[:, 1] >= 0) & (mask_ijk[:, 1] < mni_mask.shape[1]) &
             (mask_ijk[:, 2] >= 0) & (mask_ijk[:, 2] < mni_mask.shape[2]))
    mask_ijk = mask_ijk[valid]
    ijk      = ijk[valid]
    inside   = mni_mask[mask_ijk[:, 0], mask_ijk[:, 1], mask_ijk[:, 2]]
    ijk      = ijk[inside]

    out = np.zeros(mri.shape, dtype=bool)
    out[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return out


def build_atrophy_volume(mri_shape: tuple,
                         subject_affine: np.ndarray,
                         atlas_data: np.ndarray,
                         inv_atlas_affine: np.ndarray,
                         atrophy_vec: np.ndarray) -> torch.Tensor:
    """Build a (1, H, W, D) spatial atrophy map in subject/MNI space.

    Each voxel gets the atrophy z-score of its DK86 region.
    Atlas label L (1–86) → REGION_COLS index L-1 → atrophy_vec[L-1].
    Background (label 0) and out-of-atlas voxels → 0.0.

    Parameters
    ----------
    mri_shape        : (H, W, D) of the subject volume
    subject_affine   : (4,4) affine of the subject volume
    atlas_data       : (aH, aW, aD) int32
    inv_atlas_affine : (4,4)
    atrophy_vec      : (86,) float32

    Returns
    -------
    (1, H, W, D) float32 tensor
    """
    H, W, D  = mri_shape
    volume   = np.zeros((H, W, D), dtype=np.float32)

    # Build index grids for all voxels
    ii, jj, kk = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)

    # Map to atlas voxel coords
    xyz      = nib.affines.apply_affine(subject_affine, ijk)
    atlas_ijk = np.round(nib.affines.apply_affine(inv_atlas_affine, xyz)).astype(int)

    valid = ((atlas_ijk[:, 0] >= 0) & (atlas_ijk[:, 0] < atlas_data.shape[0]) &
             (atlas_ijk[:, 1] >= 0) & (atlas_ijk[:, 1] < atlas_data.shape[1]) &
             (atlas_ijk[:, 2] >= 0) & (atlas_ijk[:, 2] < atlas_data.shape[2]))

    atlas_ijk_v = atlas_ijk[valid]
    ijk_v       = ijk[valid]
    labels      = atlas_data[atlas_ijk_v[:, 0], atlas_ijk_v[:, 1], atlas_ijk_v[:, 2]]

    # Labels 1–86 → atrophy_vec index 0–85
    in_region = (labels >= 1) & (labels <= 86)
    ijk_r     = ijk_v[in_region]
    scores    = atrophy_vec[labels[in_region] - 1]
    volume[ijk_r[:, 0], ijk_r[:, 1], ijk_r[:, 2]] = scores

    return torch.from_numpy(volume).unsqueeze(0)   # (1, H, W, D)
