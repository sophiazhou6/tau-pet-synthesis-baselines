#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Evaluate the Dense-U-Net baseline with TauGenNet's canonical metrics — outputs
redirected into THIS repo.

This is a thin wrapper, NOT a fork: it runs the exact same
taugennet/scripts/evaluate_final.py the main diffusion model is scored with, in
--use-cached mode (no diffusion model, no GPU needed). Scoring the baseline with
identical metric code is the whole point — so we deliberately do not duplicate
the metric logic, which would drift over time and make the comparison unfair.

All it does is bake in baseline-appropriate defaults and absolute output paths:
  - cached predictions ← results/generated/<mode>/
  - figures           → results/figures/<mode>/   (incl. glass_brain/)
  - metrics/CSVs       → results/records/

The absolute --figures-dir works because evaluate_final.py computes
FIG_DIR = os.path.join(FIGURES_DIR, args.figures_dir), and os.path.join collapses
to the second argument when it is absolute — so the taugennet FIGURES_DIR prefix
is discarded and figures land here instead.

Usage:
    python scripts/evaluate.py --mode atrophy
    python scripts/evaluate.py --mode atrophy --fold 0      # CV fold
Unrecognized flags are forwarded verbatim to evaluate_final.py.

With --fold i, scores fold i's cached predictions and writes a per-fold
metrics.json (consumed by taugennet/scripts/aggregate_cv_eval.py for the CV
mean ± std), keeping outputs under results/{figures,records/cv_<mode>}/fold_<i>/.
"""

import argparse
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import TAUGENNET_ROOT, REPO_ROOT, GENERATED_DIR, N_FOLDS  # noqa: E402

FIGURES_DIR = os.path.join(REPO_ROOT, "results", "figures")
RECORDS_DIR = os.path.join(REPO_ROOT, "results", "records")


def main():
    p = argparse.ArgumentParser(
        description="Score the Dense-U-Net baseline with TauGenNet metrics (cached).")
    p.add_argument("--mode", choices=["atrophy", "ptau217"], required=True)
    p.add_argument("--fold", type=int, default=None,
                   help="0-indexed CV fold (default dirs: <...>/<mode>/fold_<i>/).")
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--generated-dir", default=None,
                   help="Cached predictions dir (default: results/generated/<mode>/[fold_<i>]).")
    args, extra = p.parse_known_args()

    fold_sub = "" if args.fold is None else f"fold_{args.fold}"
    gen_dir = args.generated_dir or os.path.join(GENERATED_DIR, args.mode, fold_sub)
    fig_dir = os.path.join(FIGURES_DIR, args.mode, fold_sub)  # absolute → overrides taugennet FIGURES_DIR
    # Per-fold records live under results/records/cv_<mode>/fold_<i>/ so
    # aggregate_cv_eval.py can sweep fold_*/metrics.json. Single-split → results/records/.
    rec_dir = (RECORDS_DIR if args.fold is None
               else os.path.join(RECORDS_DIR, f"cv_{args.mode}", fold_sub))
    if not os.path.isdir(gen_dir) or not any(f.endswith(".npy") for f in os.listdir(gen_dir)):
        sys.exit(f"No cached predictions in {gen_dir}. Run scripts/generate.py --mode "
                 f"{args.mode}{'' if args.fold is None else f' --fold {args.fold}'} first.")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)

    # evaluate_final.py imports `from src...` (resolved via taugennet's editable
    # install) and `from glass_brain_final import run_mode` (fallback), so put
    # taugennet root + its scripts/ on the path.
    for pth in (TAUGENNET_ROOT, os.path.join(TAUGENNET_ROOT, "scripts")):
        if pth not in sys.path:
            sys.path.insert(0, pth)

    # Per-fold: forward --fold/--n-folds (so evaluate_final builds the matching
    # test_ds) and --metrics-out (whole-brain DK86 JSON for CV aggregation).
    fold_args = ([] if args.fold is None else
                 ["--fold", str(args.fold), "--n-folds", str(args.n_folds),
                  "--metrics-out", os.path.join(rec_dir, "metrics.json")])

    sys.argv = [
        "evaluate_final.py",
        "--mode", args.mode,
        "--use-cached",
        "--generated-dir", gen_dir,
        "--figures-dir", fig_dir,
        "--records-dir", rec_dir,
        "--skip-ablation",
    ] + fold_args + extra

    tag = args.mode if args.fold is None else f"{args.mode} fold {args.fold}/{args.n_folds}"
    print(f"[baseline eval] {tag}")
    print(f"  cached  ← {gen_dir}")
    print(f"  figures → {fig_dir}")
    print(f"  records → {rec_dir}")

    eval_path = os.path.join(TAUGENNET_ROOT, "scripts", "evaluate_final.py")
    runpy.run_path(eval_path, run_name="__main__")


if __name__ == "__main__":
    main()
