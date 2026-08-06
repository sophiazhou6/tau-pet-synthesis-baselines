#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
run_cv.py — orchestrate 5-fold cross-validation training.

Usage:
  # Submit all 5 folds as a SLURM array job (recommended):
  python scripts/run_cv.py

  # Run all folds sequentially in this process (no SLURM):
  python scripts/run_cv.py --local

  # Print commands without running:
  python scripts/run_cv.py --dry-run

  # Run only specific modes:
  python scripts/run_cv.py --modes atrophy ptau217

After training, evaluate any fold with:
  python scripts/evaluate_final.py --mode atrophy \
      --checkpoint-dir results/checkpoints/cv/fold_0/atrophy \
      --arch silu --use-mask
"""
import argparse
import os
import subprocess
import sys

PYTHON = "/home/sz3962/.conda/envs/taugennet/bin/python3"
ROOT   = "/scratch/network/sz3962/taugennet"
SLURM  = os.path.join(ROOT, "slurm/train/train_cv.slurm")


def build_fold_commands(fold, n_folds, modes, ae_epochs, diff_epochs, patience):
    ae_ckpt = f"results/checkpoints/cv/fold_{fold}/taugennet_checkpoint.pt"
    cmds = []

    cmds.append([
        PYTHON, "scripts/train.py",
        "--mode", "atrophy",
        "--fold", str(fold), "--n-folds", str(n_folds),
        "--ae-epochs", str(ae_epochs), "--diff-epochs", "0",
        "--ae-checkpoint", ae_ckpt,
    ])

    for mode in modes:
        cmds.append([
            PYTHON, "scripts/train.py",
            "--mode", mode,
            "--fold", str(fold), "--n-folds", str(n_folds),
            "--skip-ae", "--ae-checkpoint", ae_ckpt,
            "--checkpoint-dir", f"results/checkpoints/cv/fold_{fold}/{mode}",
            "--diff-epochs", str(diff_epochs),
            "--patience", str(patience),
            "--val-every", "1",
            "--monitor-every", "25",
        ])

    return cmds


def parse_args():
    p = argparse.ArgumentParser(description="Run k-fold CV training for TauGenNet")
    p.add_argument("--n-folds",     type=int,  default=5)
    p.add_argument("--modes",       nargs="+", default=["atrophy", "ptau217", "combined"])
    p.add_argument("--ae-epochs",   type=int,  default=100)
    p.add_argument("--diff-epochs", type=int,  default=2000)
    p.add_argument("--patience",    type=int,  default=50)
    p.add_argument("--local",  action="store_true",
                   help="Run folds sequentially in this process (no SLURM)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.local and not args.dry_run:
        print(f"Submitting SLURM array job: folds 0–{args.n_folds - 1}")
        result = subprocess.run(
            ["sbatch", f"--array=0-{args.n_folds - 1}", SLURM],
            cwd=ROOT, capture_output=True, text=True,
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            sys.exit(result.returncode)
        print(f"\nMonitor with: squeue -u $USER")
        print(f"Logs: results/logs/taugennet_cv_<ARRAY_ID>_<FOLD>.out")
        return

    for fold in range(args.n_folds):
        cmds = build_fold_commands(
            fold, args.n_folds, args.modes,
            args.ae_epochs, args.diff_epochs, args.patience,
        )
        print(f"\n=== Fold {fold}/{args.n_folds} ===")
        for cmd in cmds:
            print("  " + " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
