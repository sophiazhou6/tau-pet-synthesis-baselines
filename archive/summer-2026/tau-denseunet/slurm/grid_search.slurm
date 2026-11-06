#!/bin/bash
#SBATCH --job-name=denseunet_grid
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-29
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/grid_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/grid_%A_%a.err

# Hyperparameter grid search across ALL 5 folds (cross-validated selection).
# Grid = lr {5e-5, 1e-4, 3e-4} x weight_decay {0, 1e-4} = 6 configs x 5 folds = 30 tasks.
# Selection: config with the lowest MEAN validation loss across its 5 folds.
# Then report THAT config's test metrics (test set never used for selection → unbiased).
#
# ⚠️  FIXED EPOCHS: trains a FIXED number of epochs (EPOCHS) with early stopping DISABLED
#     (patience = epochs), so every config/fold is compared on the same budget. Set EPOCHS
#     deliberately before submitting.
#
# array index i (0..29):  config = i / 5 ,  fold = i % 5
# Usage:  sbatch slurm/grid_search.slurm atrophy dk86
set -euo pipefail

MODE="${1:-atrophy}"
MASK="${2:-dk86}"
EPOCHS=126                   # median best-epoch across the 5 DK86 folds (47,48,126,135,152)

PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
cd "$REPO"
source $HOME/taugennet/scripts/wait_for_fac.sh
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:/home/remote/adebnath1/tau-denseunet-baseline:$HOME/taugennet:${PYTHONPATH:-}"

# --- grid definition (6 configs) ---
LRS=(5e-5 5e-5 1e-4 1e-4 3e-4 3e-4)
WDS=(0    1e-4 0    1e-4 0    1e-4)

i="${SLURM_ARRAY_TASK_ID:-0}"
CFG=$(( i / 5 ))             # 0..5  → which hyperparameter config
FOLD=$(( i % 5 ))           # 0..4  → which CV fold
LR="${LRS[$CFG]}"; WD="${WDS[$CFG]}"
TAG="grid/lr${LR}_wd${WD}"

echo "Task $i: config $CFG (lr=$LR wd=$WD) x fold $FOLD  epochs=$EPOCHS (fixed)  tag=$TAG"
mkdir -p results/logs

# Fixed epochs → patience == epochs disables early stopping (best-val checkpoint still saved).
$PYTHON scripts/train.py --mode "$MODE" --mask-mode "$MASK" --fold "$FOLD" \
    --lr "$LR" --weight-decay "$WD" --epochs "$EPOCHS" --patience "$EPOCHS" \
    --out-tag "$TAG" --overwrite

echo "Done task $i. After all 30 finish, rank configs by MEAN val loss across folds:"
echo "  $PYTHON scripts/grid_summary.py --mode $MODE --mask-mode $MASK"
