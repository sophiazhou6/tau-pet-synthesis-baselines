#!/bin/bash
#SBATCH --job-name=tau_baselines
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=adroit-h11g2
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/baselines_%A_%a.out
#SBATCH --error=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/baselines_%A_%a.err
#SBATCH --mail-type=END,FAIL

# One array task = one (model, fold): train → generate (held-out test) → score.
#
#   sbatch --array=0-24 slurm/baselines_cv.slurm <manifest> "denseunet swinunetr eagan ct2pet jointsyn"
#   sbatch --array=0-4  slurm/baselines_cv.slurm mentor235 "swinunetr"            # 5 folds of one model
#   FOLDS="0" sbatch --array=0-4 slurm/baselines_cv.slurm mentor235 "denseunet swinunetr eagan ct2pet jointsyn"
#
# array index i → model = MODELS[i / n_folds], fold = FOLDS[i % n_folds]  (FOLDS default "0 1 2 3 4").
# Wall time picks the QOS (measured 2026-09-24): ≤ 4 h → gpu-short (4 GPUs concurrently per user),
# 4–24 h → gpu-medium (2 GPUs), > 24 h → gpu-long (2). The default 03:59:00 fits denseunet /
# swinunetr / eagan; give the slower ones more, e.g.
#   sbatch --time=11:59:00 --array=0-4 slurm/baselines_cv.slurm cohort_v2 ct2pet
# EXTRA_TRAIN_ARGS="--epochs 50" passes flags to train_baseline.py; SCORE_ARGS="--use-controls-322 …" to score_baseline.py. adroit-h11g2 (20 GB MIG
# slices) is excluded: the 3D GANs at 128³ need a full A100/A40.
set -euo pipefail

MANIFEST="${1:?manifest name or path}"
MODELS_STR="${2:-denseunet swinunetr eagan ct2pet jointsyn}"
read -r -a MODELS <<< "$MODELS_STR"
read -r -a FOLDS <<< "${FOLDS:-0 1 2 3 4}"
NF=${#FOLDS[@]}
i="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$(( i / NF ))]}"
FOLD="${FOLDS[$(( i % NF ))]}"
TAG="${TAG:-$MODEL}"
# FOLDS="none" → single split without --fold (with EXTRA_TRAIN_ARGS="--val-frac 0": train on the whole dev pool)
FOLD_ARG="--fold $FOLD"; [ "$FOLD" = "none" ] && FOLD_ARG=""
# STEPS picks which stages run (default all four); e.g. STEPS="gen score val" finishes a run whose
# training completed (uses its best.pt), STEPS="val" only writes the validation-set predictions.
STEPS=" ${STEPS:-train gen score val} "
step() { [[ "$STEPS" == *" $1 "* ]]; }

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
REPO=/scratch/network/sz3962/tau-denseunet-baseline
export TAUGENNET_ROOT="${TAUGENNET_ROOT:-/scratch/network/sz3962/taugennet}"
cd "$REPO"; mkdir -p results/logs

echo "task $i → model=$MODEL tag=$TAG fold=$FOLD manifest=$MANIFEST  ($(date))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Quota guard: a hard-quota hit on /scratch/network kills every running job of this account, so
# refuse to start when usage is already above QUOTA_MAX_GIB (default 185 of the 195/200 GiB limit).
USED=$( (checkquota 2>/dev/null || true) | awk '$0 ~ "/scratch/network" && $0 ~ "GiB" {v=$5; sub("GiB","",v); print v; exit}')
if [ -n "${USED:-}" ]; then
  if awk -v u="$USED" -v m="${QUOTA_MAX_GIB:-185}" 'BEGIN{exit !(u>m)}'; then
    echo "ABORT: /scratch/network usage is ${USED} GiB (> ${QUOTA_MAX_GIB:-185}). Free space, then resubmit."; exit 3
  fi
  echo "scratch usage: ${USED} GiB (guard at ${QUOTA_MAX_GIB:-185})"
else
  echo "warning: could not read checkquota; continuing without the quota guard"
fi

if step train; then
  echo "── 1/4 train ──"
  $PYTHON scripts/train_baseline.py --model "$MODEL" --tag "$TAG" --manifest "$MANIFEST" $FOLD_ARG \
      ${EXTRA_TRAIN_ARGS:-}
else
  echo "── 1/4 train: skipped (STEPS=$STEPS) ──"
fi
if step gen; then
  echo "── 2/4 generate (held-out test) ──"
  $PYTHON scripts/generate_baseline.py --model "$MODEL" --tag "$TAG" --manifest "$MANIFEST" $FOLD_ARG --overwrite
else
  echo "── 2/4 generate (held-out test): skipped (STEPS=$STEPS) ──"
fi
if step score; then
  echo "── 3/4 score ──"
  $PYTHON scripts/score_baseline.py --model "$MODEL" --tag "$TAG" --manifest "$MANIFEST" $FOLD_ARG ${SCORE_ARGS:-}
else
  echo "── 3/4 score: skipped (STEPS=$STEPS) ──"
fi
if step val; then
  echo "── 4/4 validation-set predictions (kept for the downstream AD/MCI/CN classifier) ──"
  $PYTHON scripts/generate_baseline.py --model "$MODEL" --tag "$TAG" --manifest "$MANIFEST" $FOLD_ARG \
      --split val --overwrite || echo "WARN: validation-set generation failed (core results unaffected)"
  # pack THIS run's val predictions (lossless, ~11x smaller) so loose .npy never pile up; only this
  # fold's directory: other folds of the same model may still be writing theirs
  $PYTHON - "$MANIFEST" "$TAG" "$FOLD" <<'PACKPY' || echo "WARN: val-pack failed (prune_checkpoints will catch it)"
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "src")); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from denseunet.config import GEN_ROOT
from denseunet import predictions as preds
from train_baseline import resolve_manifest, run_dir
manifest, tag, fold = sys.argv[1], sys.argv[2], sys.argv[3]
d = run_dir(GEN_ROOT, resolve_manifest(manifest), tag, None if fold == "none" else int(fold)) + "_val"
if preds.npy_files(d):
    preds.pack(d)
PACKPY
else
  echo "── 4/4 validation-set predictions (kept for the downstream AD/MCI/CN classifier): skipped (STEPS=$STEPS) ──"
fi
echo "done ($(date)). Aggregate all folds/models with:"
echo "  $PYTHON scripts/aggregate_baselines.py --cohort $(basename "${MANIFEST%.csv}")"
