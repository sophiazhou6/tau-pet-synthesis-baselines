#!/bin/bash
#SBATCH --job-name=denseunet_cv
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/denseunet_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/denseunet_%A_%a.err

# 5-fold CV for the Dense-U-Net baseline — one SLURM array task per fold.
# Per-fold pipeline (no inter-step delay):
#   1. train     (scripts/train.py     --fold $FOLD --mask-mode $MASK)
#   2. generate  (scripts/generate.py  --fold $FOLD --mask-mode $MASK)  → fixed 47 test
#   3. score     dk86:  scripts/evaluate.py (rich figures + metrics via evaluate_final)
#   4. SUVR sets scripts/suvr_sets.py  --fold $FOLD --mask-mode $MASK
#                → metrics.json + SET 1/2/3 (SUVR) + figures
#
# The mentor split fixes the 47 heldout test for every fold; array tasks rotate the
# validation fold over the 188 dev subjects.
#
# Usage:  sbatch slurm/train_generate.slurm atrophy              # DK86 mask (default)
#         sbatch slurm/train_generate.slurm atrophy wholebrain   # whole-brain (T1-seg) mask
#
# Aggregate after all 5 folds:
#   python /home/remote/adebnath1/taugennet/scripts/aggregate_cv_eval.py results/records/cv_<run_tag>
#   python scripts/aggregate_suvr_sets.py --mode <mode> --mask-mode <MASK>
set -euo pipefail

MODE="${1:-atrophy}"
MASK="${2:-dk86}"                       # dk86 | wholebrain
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
RUN_TAG="$MODE"; [ "$MASK" = "wholebrain" ] && RUN_TAG="${MODE}_wholebrain"
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet

cd "$REPO"
source $HOME/taugennet/scripts/wait_for_fac.sh
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:/home/remote/adebnath1/tau-denseunet-baseline:$HOME/taugennet:${PYTHONPATH:-}"
mkdir -p results/logs "results/checkpoints/$RUN_TAG/fold_$FOLD" \
         "results/generated/$RUN_TAG/fold_$FOLD" "results/records/cv_$RUN_TAG/fold_$FOLD"

echo "Job:         ${SLURM_ARRAY_JOB_ID:-interactive} task ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Mode/Mask/Fold: $MODE / $MASK / fold $FOLD   (run_tag=$RUN_TAG)"
echo "Start time:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo ""

echo "── 1/4 Training (fold $FOLD, mask $MASK) ─────────────────────────────────"
$PYTHON scripts/train.py --mode "$MODE" --fold "$FOLD" --mask-mode "$MASK"

echo "── 2/4 Generating cached test predictions (fold $FOLD) ───────────────────"
$PYTHON scripts/generate.py --mode "$MODE" --fold "$FOLD" --mask-mode "$MASK"

if [ "$MASK" = "dk86" ]; then
  echo "── 3/4 Scoring with TauGenNet evaluate_final (rich figures, fold $FOLD) ──"
  # evaluate_final.py is DK86-hardcoded → only run it for the dk86 variant.
  $PYTHON scripts/evaluate.py --mode "$MODE" --fold "$FOLD"
else
  echo "── 3/4 (skipped for wholebrain: evaluate_final.py is DK86-only) ──────────"
fi

echo "── 4/4 Whole-brain metrics + SUVR SET 1/2/3 (fold $FOLD, mask $MASK) ──────"
$PYTHON scripts/suvr_sets.py --mode "$MODE" --fold "$FOLD" --mask-mode "$MASK"

echo ""
echo "Fold $FOLD done ($RUN_TAG). After all 5 folds, aggregate with:"
echo "  $PYTHON $TAUGENNET_ROOT/scripts/aggregate_cv_eval.py $REPO/results/records/cv_$RUN_TAG"
echo "  $PYTHON scripts/aggregate_suvr_sets.py --mode $MODE --mask-mode $MASK"
