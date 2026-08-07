#!/bin/bash
#SBATCH --job-name=denseunet_tuned_eval
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/tuned_eval_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/tuned_eval_%A_%a.err

# Unbiased test-set eval of the WINNING grid-search config (picked by validation loss).
# Reuses existing grid-search checkpoints — does NOT retrain.
# Usage: sbatch slurm/eval_tuned_grid.slurm atrophy dk86 lr1e-4_wd0
set -euo pipefail

MODE="${1:-atrophy}"
MASK="${2:-dk86}"
CFG_TAG="${3:?Must pass the winning grid config tag, e.g. lr1e-4_wd0}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet

CKPT="$REPO/results/checkpoints/$MODE/grid/$CFG_TAG/fold_$FOLD/denseunet_best.pt"
GEN_DIR="$REPO/results/generated/${MODE}_tuned_${CFG_TAG}/fold_$FOLD"

cd "$REPO"
source "$HOME/taugennet/scripts/wait_for_fac.sh"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$REPO:$HOME/taugennet:${PYTHONPATH:-}"
mkdir -p results/logs "$GEN_DIR"

echo "Job: ${SLURM_ARRAY_JOB_ID:-interactive} task ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Mode/Mask/Fold/Config: $MODE / $MASK / fold $FOLD / $CFG_TAG"
echo "Checkpoint: $CKPT"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "── 1/3 Generate test predictions from tuned checkpoint (fold $FOLD) ──────"
$PYTHON scripts/generate.py --mode "$MODE" --fold "$FOLD" --mask-mode "$MASK" \
    --checkpoint "$CKPT" --out-dir "$GEN_DIR"

if [ "$MASK" = "dk86" ]; then
  echo "── 2/3 Scoring with TauGenNet evaluate_final (fold $FOLD) ────────────────"
  $PYTHON scripts/evaluate.py --mode "$MODE" --fold "$FOLD" --generated-dir "$GEN_DIR"
else
  echo "── 2/3 (skipped for wholebrain: evaluate_final.py is DK86-only) ──────────"
fi

echo "── 3/3 SUVR SET 1/2/3 (fold $FOLD, mask $MASK) ────────────────────────────"
$PYTHON scripts/suvr_sets.py --mode "$MODE" --fold "$FOLD" --mask-mode "$MASK" --generated-dir "$GEN_DIR"

echo "Fold $FOLD done ($CFG_TAG)."
