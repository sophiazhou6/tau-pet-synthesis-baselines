#!/bin/bash
#SBATCH --job-name=rerank_valeval
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-24
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/rerank_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/rerank_%A_%a.err
set -euo pipefail
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
cd "$REPO"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$REPO:$HOME/taugennet:${PYTHONPATH:-}"
mkdir -p results/logs

CONFIGS=(lr1e-4_wd1e-4 lr3e-4_wd0 lr3e-4_wd1e-4 lr5e-5_wd0 lr5e-5_wd1e-4)
CFG_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
CONFIG=${CONFIGS[$CFG_IDX]}

CKPT="$REPO/results/checkpoints/atrophy/grid/${CONFIG}/fold_${FOLD}/denseunet_best.pt"
GEN_DIR="$REPO/results/generated/atrophy_grid_${CONFIG}_valeval/fold_${FOLD}"
mkdir -p "$GEN_DIR"

echo "-- config=$CONFIG fold=$FOLD --"
$PYTHON scripts/generate.py --mode atrophy --mask-mode dk86 --fold "$FOLD" \
    --checkpoint "$CKPT" --out-dir "$GEN_DIR" --split val

$PYTHON scripts/suvr_sets.py --mode atrophy --fold "$FOLD" --mask-mode dk86 \
    --generated-dir "$GEN_DIR" --out-tag "grid_${CONFIG}_valeval" --split val

echo "-- Publishing to FAC --"
bash "$HOME/taugennet/scripts/publish_to_fac.sh"
