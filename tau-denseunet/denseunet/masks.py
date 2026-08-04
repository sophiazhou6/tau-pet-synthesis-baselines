"""
Whole-brain mask for the "no-DK86" baseline variant.

The default baseline masks to the 86-region Desikan-Killiany atlas (~8.7% of the
volume). This module builds an alternative **whole-brain** mask that excludes only
background, derived from the T1 tissue segmentation (T1_seg_in_MNI.nii.gz, labels
0=background / 1,2,3=tissue) — i.e. "the T1 seg excluding label 0" (~25.5%).

Because evaluate_final.py and the metric functions use ONE shared mask for all
subjects (so cross-subject coverage is identical, exactly like the DK86 atlas
mask), we build a single **consensus** MNI-space mask: a voxel is in-brain if
seg>0 in at least `thresh` of the cohort's subjects. The mask is cached to
results/masks/wholebrain_<mode>.npy and injected into the dataset's shared mask
slot (`_dk_mask`), so training, generation, and scoring all operate on it.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from .config import REPO_ROOT, TAUGENNET_ROOT, VOL_SHAPE, build_dataloaders

MASK_DIR = os.path.join(REPO_ROOT, "results", "masks")


def _seg_path(mri_path):
    # Mirrors taugennet/src/dataset_spatial.SpatialTauPETDataset._seg_path
    return mri_path.replace("T1_to_MNI_nonlin.nii.gz", "T1_seg_in_MNI.nii.gz")


def wholebrain_mask_path(mode):
    return os.path.join(MASK_DIR, f"wholebrain_{mode}.npy")


def build_wholebrain_mask(mode, thresh=0.5, save=True):
    """Consensus whole-brain mask (1,H,W,D) float tensor from the cohort's T1 segs.

    A voxel is in-brain if (seg>0) in >= `thresh` fraction of subjects. Built over
    every matched subject in the mode (all folds share the same mentor cohort).
    """
    import nibabel as nib

    # All matched MRI paths (union of the three sets = the full mentor cohort).
    tr, va, te, *_ = build_dataloaders(mode=mode, use_dk_mask=False)
    mri_paths = list(dict.fromkeys(tr.mri_paths + va.mri_paths + te.mri_paths))

    acc = np.zeros(VOL_SHAPE, dtype=np.float32)
    n = 0
    for mri in mri_paths:
        seg = _seg_path(mri)
        if not os.path.exists(seg):
            continue
        vol = nib.load(seg).get_fdata().astype(np.float32)
        vt = F.interpolate(torch.from_numpy(vol)[None, None], size=VOL_SHAPE,
                           mode="nearest").squeeze().numpy()
        acc += (vt > 0).astype(np.float32)
        n += 1
    if n == 0:
        raise RuntimeError(f"No T1 seg files found for mode={mode}.")

    consensus = (acc / n) >= thresh                       # (H,W,D) bool
    mask = torch.from_numpy(consensus.astype(np.float32))[None]  # (1,H,W,D)
    if save:
        os.makedirs(MASK_DIR, exist_ok=True)
        np.save(wholebrain_mask_path(mode), mask.numpy())
    print(f"[wholebrain mask] {mode}: {int(consensus.sum())} voxels "
          f"({100 * consensus.mean():.1f}% of volume) from {n} subjects, thresh>={thresh}")
    return mask


def load_or_build_wholebrain_mask(mode):
    p = wholebrain_mask_path(mode)
    if os.path.exists(p):
        return torch.from_numpy(np.load(p))
    return build_wholebrain_mask(mode)


def inject_mask(mask, *datasets):
    """Replace each dataset's shared mask slot with `mask` (moved to CPU float).

    After this, __getitem__ masks the volumes to `mask`, and evaluate_final's
    _dk86_mask(ds) (which reads ds._dk_mask) returns it too — so the whole pipeline
    uses the whole-brain region instead of DK86. Call right after build_dataloaders,
    before any sample is accessed.
    """
    m = mask.float()
    for ds in datasets:
        if ds is not None:
            ds._dk_mask = m
    return m
