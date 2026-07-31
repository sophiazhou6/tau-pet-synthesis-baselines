#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
evaluate_tissue.py — Tissue-stratified evaluation for TauGenNet.

Loads pre-generated volumes (--use-cached) and computes per-tissue metrics
using T1_seg_in_MNI.nii.gz segmentation files (labels 1=GM, 2=WM, 3=CSF).

Usage:
  python scripts/evaluate_tissue.py \\
      --mode atrophy \\
      --generated-dir results/generated/small/atrophy \\
      --records-dir results/records/small
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import VOL_SHAPE
from src import dataset_final as _dataset_v2

TISSUE_LABELS = {1: "GM", 2: "WM", 3: "CSF"}
METRICS = ["pearson", "nrmse", "ssim", "mse", "mae"]


def _safe_ssim(r, g, data_range=1.0):
    win = min(7, min(r.shape))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return float("nan")
    return float(ssim(r, g, data_range=data_range, win_size=win))


def _ssim_tissue(real, gen, tissue_mask, data_range=1.0):
    """SSIM on bounding box of the tissue mask."""
    nz = np.argwhere(tissue_mask)
    if nz.size == 0:
        return float("nan")
    lo, hi = nz.min(0), nz.max(0) + 1
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return _safe_ssim(real[sl], gen[sl], data_range=data_range)


def load_seg(seg_path):
    """Load T1_seg_in_MNI.nii.gz and resample to VOL_SHAPE using nearest-neighbor."""
    vol = nib.load(seg_path).get_fdata().astype(np.float32)
    t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    resampled = F.interpolate(t, size=VOL_SHAPE, mode="nearest").squeeze().numpy()
    return resampled.astype(np.int32)


def load_cached_generations(test_ds, gen_dir):
    """Load pre-generated .npy volumes from disk. Returns parallel lists."""
    real_list, gen_list, indices = [], [], []
    n_missing = 0
    for i in tqdm(range(len(test_ds)), desc="Loading cached"):
        gen_path = os.path.join(gen_dir, f"subject_{i:03d}.npy")
        if not os.path.exists(gen_path):
            n_missing += 1
            continue
        pet, _, _ = test_ds[i]
        real = pet.squeeze().numpy()
        gen  = np.load(gen_path).astype(np.float32)
        real_list.append(real)
        gen_list.append(gen * (real > 0))
        indices.append(i)
    if not real_list:
        raise FileNotFoundError(f"No subject_*.npy found in {gen_dir}")
    if n_missing:
        print(f"  WARNING: {n_missing} subject(s) missing from cache — "
              f"loaded {len(real_list)}/{len(test_ds)}")
    return real_list, gen_list, indices


def compute_tissue_metrics(real_list, gen_list, indices, test_ds):
    """Per-subject per-tissue metrics. Returns {label: {metric: [per-subject values]}}."""
    results = {lbl: {m: [] for m in METRICS} for lbl in TISSUE_LABELS}
    n_no_seg = 0

    for real, gen, subj_idx in tqdm(
            zip(real_list, gen_list, indices), total=len(real_list),
            desc="Computing tissue metrics"):
        mri_path = test_ds.mri_paths[subj_idx]
        seg_path = mri_path.replace("T1_to_MNI_nonlin.nii.gz", "T1_seg_in_MNI.nii.gz")

        if not os.path.exists(seg_path):
            n_no_seg += 1
            for lbl in TISSUE_LABELS:
                for m in METRICS:
                    results[lbl][m].append(float("nan"))
            continue

        seg = load_seg(seg_path)

        for lbl, name in TISSUE_LABELS.items():
            tissue_mask = (seg == lbl) & (real > 0)
            if tissue_mask.sum() < 10:
                for m in METRICS:
                    results[lbl][m].append(float("nan"))
                continue

            r_in = real[tissue_mask]
            g_in = gen[tissue_mask]
            diff = g_in - r_in
            mse   = float(np.mean(diff ** 2))
            mae   = float(np.mean(np.abs(diff)))
            nrmse = float(np.sqrt(mse) / (r_in.max() - r_in.min() + 1e-8))

            if r_in.std() < 1e-8 or g_in.std() < 1e-8:
                rho = float("nan")
            else:
                rho = float(pearsonr(r_in, g_in)[0])

            ssim_score = _ssim_tissue(real, gen, tissue_mask)

            results[lbl]["pearson"].append(rho)
            results[lbl]["nrmse"].append(nrmse)
            results[lbl]["ssim"].append(ssim_score)
            results[lbl]["mse"].append(mse)
            results[lbl]["mae"].append(mae)

    if n_no_seg:
        print(f"  WARNING: {n_no_seg} subject(s) had no seg file — NaN inserted.")

    return results


def summarize_and_print(results):
    """Print mean ± std per tissue × metric and return a summary DataFrame."""
    rows = []
    for lbl, name in TISSUE_LABELS.items():
        row = {"Tissue": name}
        for m in METRICS:
            vals = np.asarray([v for v in results[lbl][m] if not np.isnan(v)])
            mean = float(vals.mean()) if len(vals) > 0 else float("nan")
            std  = float(vals.std())  if len(vals) > 0 else float("nan")
            row[f"{m}_mean"] = mean
            row[f"{m}_std"]  = std
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Tissue")

    print("\nTissue-stratified metrics (mean ± std, per-subject, in-brain voxels):")
    for m in METRICS:
        print(f"\n  {m.upper()}")
        for name in ["GM", "WM", "CSF"]:
            mean = df.loc[name, f"{m}_mean"]
            std  = df.loc[name, f"{m}_std"]
            print(f"    {name}: {mean:.4f} ± {std:.4f}")
    return df


def save_outputs(results, summary_df, mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    md_lines = [f"# Tissue-Stratified Evaluation — {mode} mode\n",
                "\n## Summary (mean ± std per subject)\n"]

    for m in METRICS:
        tbl = pd.DataFrame(
            {name: {
                "mean": summary_df.loc[name, f"{m}_mean"],
                "std":  summary_df.loc[name, f"{m}_std"],
            } for name in ["GM", "WM", "CSF"]}
        ).T
        md_lines.append(f"\n### {m.upper()}\n\n")
        md_lines.append(tbl.to_markdown(floatfmt=".4f") + "\n")

        for lbl, name in TISSUE_LABELS.items():
            vals = results[lbl][m]
            df_per_subj = pd.DataFrame({"value": vals})
            csv_path = os.path.join(out_dir, f"tissue_{name.lower()}_{m}_{mode}.csv")
            df_per_subj.to_csv(csv_path)

    md_path = os.path.join(out_dir, f"tissue_metrics_{mode}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nTables → {md_path}")
    print(f"CSVs   → {out_dir}/tissue_*_{mode}.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tissue-stratified TauGenNet evaluation")
    p.add_argument("--mode", choices=["atrophy", "ptau217"], required=True)
    p.add_argument("--generated-dir", type=str, required=True,
                   help="Directory containing subject_*.npy generated volumes")
    p.add_argument("--records-dir", type=str, default="results/records",
                   help="Output directory for CSVs and markdown")
    p.add_argument("--use-mask", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply DK atlas mask when loading dataset (default: on)")
    args = p.parse_args()

    print(f"Mode:          {args.mode}")
    print(f"Generated dir: {args.generated_dir}")
    print(f"Records dir:   {args.records_dir}")

    _, _, test_ds, *_ = _dataset_v2.build_dataloaders(
        mode=args.mode, use_dk_mask=args.use_mask)
    print(f"Test subjects: {len(test_ds)}")

    real_list, gen_list, indices = load_cached_generations(test_ds, args.generated_dir)
    print(f"Loaded {len(real_list)} cached volumes.")

    results = compute_tissue_metrics(real_list, gen_list, indices, test_ds)
    summary_df = summarize_and_print(results)
    save_outputs(results, summary_df, args.mode, args.records_dir)

    print("\nDone.")
