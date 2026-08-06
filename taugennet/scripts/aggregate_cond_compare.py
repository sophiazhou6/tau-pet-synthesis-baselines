#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
aggregate_cond_compare.py — turn the cond_compare 5-fold CV runs into one comparison table.

The conditioning ablation trains, per variant, 5 diffusion models that all share a FIXED
20%% test set. dataset_spatial's CV (build_dataloaders(fold_idx=...)) rotates only the
*validation* subset within the first 80%% — `test_ds = last 20%%` is identical for every
fold (see src/dataset_spatial.py: the `if fold_idx is not None` branch). So the honest
aggregation is the **mean ± std of test metrics ACROSS the 5 fold-models** per variant — a
robustness estimate on the shared test set — NOT a pooling of disjoint held-out subjects.

Pipeline, per (variant, fold):
  1. Generate — reuse scripts/generate_spatial.py via subprocess (it derives cond_mode /
     tissue / EMA weights from the checkpoint and runs the spatial DDPM loop), caching
     subject_NNN.npy into an isolated per-(variant,fold) directory.
  2. Score — build the SAME test_ds, load the cached generations, and compute per-subject
     in-DK86 metrics, reusing evaluate_final.py's functions (no metric is reimplemented):
       load_cached_generations, _dk86_mask, _mask_bbox,
       compute_wholebrain_metrics (per-subject), compute_pooled_wholebrain_metrics (global).

Outputs (per-subject vs global kept separate; in-brain DK86-masked throughout):
  <records-dir>/per_subject_metrics.csv   — one row per (variant, fold, subject)
  <records-dir>/fold_summary.csv          — one row per (variant, fold): per-subject means + pooled
  <records-dir>/summary.md                — variant comparison: mean ± std across folds

Generation needs a GPU and is the slow part (~N_test × 500 DDPM steps × folds × variants),
so run this on a GPU node (see slurm/eval/aggregate_cond_compare.slurm). Cached subject_NNN.npy
are reused automatically; pass --regenerate to force, or --no-generate to score caches only.

Usage:
  python scripts/aggregate_cond_compare.py \
      --variants spatialmap_none:none spatialmap_ptau:ptau217 \
      --root results/checkpoints/cond_compare \
      --records-dir results/records/cond_compare
  python scripts/aggregate_cond_compare.py --variants spatialmap_none:none --dry-run
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# Make `import src...` (needed by evaluate_final) and `import evaluate_final` both resolve,
# regardless of where python is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import evaluate_final as ef          # noqa: E402  (reuse canonical eval functions)
from src import dataset_spatial as _dataset  # noqa: E402

PYTHON_DEFAULT = "/home/sz3962/.conda/envs/taugennet/bin/python3"
GEN_SCRIPT     = os.path.join(ROOT, "scripts", "generate_spatial.py")

# Metric keys as returned (in order) by compute_wholebrain_metrics.
METRIC_KEYS = ["pearson", "nrmse", "ssim", "mse", "mae"]
# Arrow direction for the summary header (↑ = higher better, ↓ = lower better).
METRIC_ARROW = {"ssim": "↑", "pearson": "↑", "nrmse": "↓", "mse": "↓", "mae": "↓"}
# Display order in the summary table.
SUMMARY_ORDER = ["ssim", "pearson", "nrmse", "mse", "mae"]


def parse_variants(items):
    """['spatialmap_none:none', 'atrophy:atrophy'] -> [('spatialmap_none','none'), ...]."""
    out = []
    for it in items:
        if ":" not in it:
            raise SystemExit(f"--variants entry must be 'label:cond_mode', got '{it}'")
        label, mode = it.split(":", 1)
        if mode not in ("none", "ptau217", "atrophy"):
            raise SystemExit(f"cond_mode must be none|ptau217|atrophy, got '{mode}'")
        out.append((label, mode))
    return out


def find_checkpoint(fold_dir, ckpt_name):
    """Prefer the *_best.pt checkpoint; fall back to the last-epoch one."""
    best = os.path.join(fold_dir, ckpt_name)
    if os.path.exists(best):
        return best
    alt = os.path.join(fold_dir, ckpt_name.replace("_best.pt", ".pt"))
    return alt if os.path.exists(alt) else None


def n_cached(gen_dir):
    return len(glob.glob(os.path.join(gen_dir, "subject_*.npy")))


def run_generation(python_bin, ckpt, gen_dir, fold, n_folds, args):
    cmd = [
        python_bin, GEN_SCRIPT,
        "--checkpoint", ckpt,
        "--generated-dir", gen_dir,
        "--unet-channels", args.unet_channels,
        "--n-transformer", str(args.n_transformer),
        "--n-steps", str(args.n_steps),
        "--fold", str(fold), "--n-folds", str(n_folds),
    ]
    print(f"    $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def _ckpt_cond_mode(ckpt_path, fallback):
    """The cond_mode the fold was actually trained with (authoritative source)."""
    import torch
    cm = torch.load(ckpt_path, map_location="cpu").get("cond_mode", fallback)
    if cm != fallback:
        print(f"    note: checkpoint cond_mode='{cm}' (CLI said '{fallback}'); using checkpoint",
              flush=True)
    return cm


def _nanmean(vals):
    a = np.asarray(vals, dtype=float)
    return float(np.nanmean(a)) if a.size and not np.all(np.isnan(a)) else float("nan")


def score_fold(label, cond_mode, fold, ckpt, gen_dir, test_ds_cache):
    """Generate-loaded per-subject metrics for one (variant, fold). Returns
    (per_subject_rows, fold_summary_row) or (None, None) if the cache is unusable."""
    # test_ds is fold-independent for a given cond_mode -> build once and reuse.
    if cond_mode not in test_ds_cache:
        _, _, test_ds, _, _, _ = _dataset.build_dataloaders(
            use_dk_mask=True, fold_idx=0, n_folds=args.n_folds, cond_mode=cond_mode,
        )
        test_ds_cache[cond_mode] = test_ds
    test_ds = test_ds_cache[cond_mode]

    try:
        real_norm, gen_masked, *_ = ef.load_cached_generations(test_ds, cond_mode, gen_dir)
    except FileNotFoundError as e:
        print(f"    SKIP fold {fold}: {e}", flush=True)
        return None, None

    mask = ef._dk86_mask(test_ds)
    bbox = ef._mask_bbox(mask) if mask is not None else None
    pe, nr, ss, ms, ma = ef.compute_wholebrain_metrics(real_norm, gen_masked, mask, bbox)

    per_subject = []
    for i, (p, n_, s, m, a) in enumerate(zip(pe, nr, ss, ms, ma)):
        per_subject.append(dict(variant=label, fold=fold, subject=i,
                                pearson=p, nrmse=n_, ssim=s, mse=m, mae=a))

    # Pooled / global voxel-level metrics for this fold (SET 1).
    pooled = ef.compute_pooled_wholebrain_metrics(real_norm, gen_masked, mask)
    pooled_row = pooled.iloc[0].to_dict()

    fold_row = dict(variant=label, fold=fold, n_test=len(real_norm))
    for k, vals in zip(METRIC_KEYS, (pe, nr, ss, ms, ma)):
        fold_row[f"{k}_persubj"] = _nanmean(vals)
    for k in METRIC_KEYS:
        if k in pooled_row:
            fold_row[f"{k}_pooled"] = float(pooled_row[k])
    return per_subject, fold_row


def build_summary_md(fold_df, variants):
    """Variant comparison table: per-subject mean ± std across folds, then a pooled block."""
    lines = []
    lines.append("# cond_compare — conditioning ablation (5-fold CV)\n")
    lines.append(
        "All folds share a **fixed 20% test set** (dataset_spatial CV rotates only the "
        "validation split). Values below are **mean ± std across the fold-models** on that "
        "shared test set. Metrics are in-brain (DK86-masked), normalized [0,1].\n")

    def block(metric_suffix, title):
        lines.append(f"\n## {title}\n")
        hdr = "| Variant | folds | N test | " + " | ".join(
            f"{k.upper()} {METRIC_ARROW[k]}" for k in SUMMARY_ORDER) + " |"
        sep = "|" + "---|" * (3 + len(SUMMARY_ORDER))
        lines.append(hdr)
        lines.append(sep)
        for label, _mode in variants:
            sub = fold_df[fold_df.variant == label]
            if sub.empty:
                lines.append(f"| {label} | 0 | – | " + " | ".join(["–"] * len(SUMMARY_ORDER)) + " |")
                continue
            n_test = int(sub.n_test.iloc[0])
            cells = []
            for k in SUMMARY_ORDER:
                col = f"{k}_{metric_suffix}"
                v = sub[col].to_numpy(dtype=float)
                cells.append(f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")
            lines.append(f"| {label} | {len(sub)} | {n_test} | " + " | ".join(cells) + " |")

    block("persubj", "Per-subject (mean over test subjects, then mean ± std across folds)")
    block("pooled",  "Pooled voxel-level (per fold, then mean ± std across folds)")
    lines.append("\n_N test differs by variant: the ptau217 variant drops subjects without a "
                 "plasma p-tau217 value._\n")
    return "\n".join(lines)


def _write(path, write_fn, force):
    if os.path.exists(path) and not force:
        raise SystemExit(f"refusing to overwrite existing {path} (pass --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_fn(path)
    print(f"wrote {path}", flush=True)


def main():
    global args
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", required=True,
                   help="label:cond_mode entries, e.g. spatialmap_none:none spatialmap_ptau:ptau217")
    p.add_argument("--root", default="results/checkpoints/cond_compare",
                   help="Dir holding <label>/fold_<k>/ checkpoint subdirs.")
    p.add_argument("--generated-root", default="results/generated/cond_compare",
                   help="Where cached subject_NNN.npy go (isolated per variant+fold).")
    p.add_argument("--records-dir", default="results/records/cond_compare")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--ckpt-name", default="diff_spatial_atrophy_best.pt")
    p.add_argument("--unet-channels", default="128,256,512")
    p.add_argument("--n-transformer", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--regenerate", action="store_true",
                   help="Re-run generation even if a cache already exists.")
    p.add_argument("--no-generate", action="store_true",
                   help="Only score existing caches; never call generate_spatial.py.")
    p.add_argument("--python", default=PYTHON_DEFAULT, help="Interpreter for generation subprocess.")
    p.add_argument("--dry-run", action="store_true",
                   help="List checkpoints found/missing and planned actions, then exit.")
    p.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    args = p.parse_args()

    variants = parse_variants(args.variants)
    root = os.path.join(ROOT, args.root) if not os.path.isabs(args.root) else args.root
    gen_root = os.path.join(ROOT, args.generated_root) if not os.path.isabs(args.generated_root) \
        else args.generated_root

    per_subject_rows, fold_rows = [], []
    test_ds_cache = {}

    for label, cond_mode in variants:
        print(f"\n=== variant {label} (cond_mode={cond_mode}) ===", flush=True)
        for k in range(args.n_folds):
            fold_dir = os.path.join(root, label, f"fold_{k}")
            ckpt = find_checkpoint(fold_dir, args.ckpt_name)
            gen_dir = os.path.join(gen_root, label, f"fold_{k}")
            if ckpt is None:
                print(f"  fold {k}: no checkpoint under {fold_dir} — skipping", flush=True)
                continue
            cached = n_cached(gen_dir)
            need_gen = args.regenerate or cached == 0
            print(f"  fold {k}: ckpt={os.path.relpath(ckpt, ROOT)}  cached={cached}  "
                  f"{'GENERATE' if (need_gen and not args.no_generate) else 'use cache'}",
                  flush=True)
            if args.dry_run:
                continue
            if need_gen and not args.no_generate:
                run_generation(args.python, ckpt, gen_dir, k, args.n_folds, args)
            elif need_gen and args.no_generate:
                print(f"    SKIP fold {k}: --no-generate and no cache in {gen_dir}", flush=True)
                continue

            true_mode = _ckpt_cond_mode(ckpt, cond_mode)
            ps, fr = score_fold(label, true_mode, k, ckpt, gen_dir, test_ds_cache)
            if ps is not None:
                per_subject_rows.extend(ps)
                fold_rows.append(fr)

    if args.dry_run:
        print("\n(dry run — no generation, scoring, or writes performed)")
        return
    if not fold_rows:
        raise SystemExit("No folds scored — nothing to aggregate (checkpoints/caches missing?).")

    ps_df = pd.DataFrame(per_subject_rows)
    fold_df = pd.DataFrame(fold_rows)

    _write(os.path.join(ROOT, args.records_dir, "per_subject_metrics.csv"),
           lambda path: ps_df.to_csv(path, index=False), args.force)
    _write(os.path.join(ROOT, args.records_dir, "fold_summary.csv"),
           lambda path: fold_df.to_csv(path, index=False), args.force)
    summary = build_summary_md(fold_df, variants)
    _write(os.path.join(ROOT, args.records_dir, "summary.md"),
           lambda path: open(path, "w").write(summary), args.force)

    print("\n" + summary)


if __name__ == "__main__":
    main()
