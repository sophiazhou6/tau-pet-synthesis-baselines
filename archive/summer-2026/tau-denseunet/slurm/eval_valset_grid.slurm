#!/bin/bash
#SBATCH --job-name=denseunet_valeval
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-4
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/valeval_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/valeval_%A_%a.err
set -euo pipefail
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
CKPT="$REPO/results/checkpoints/atrophy/grid/lr1e-4_wd0/fold_$FOLD/denseunet_best.pt"
GEN_DIR="$REPO/results/generated/atrophy_tuned_lr1e-4_wd0_valeval/fold_$FOLD"
cd "$REPO"
# source "$HOME/taugennet/scripts/wait_for_fac.sh"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$REPO:$HOME/taugennet:${PYTHONPATH:-}"
mkdir -p results/logs "$GEN_DIR"

echo "── Generate (val split, fold $FOLD) ──"
$PYTHON scripts/generate.py --mode atrophy --mask-mode dk86 --fold "$FOLD" \
    --checkpoint "$CKPT" --out-dir "$GEN_DIR" --split val

echo "── SET 1/2/3 on val split ──"
$PYTHON scripts/suvr_sets.py --mode atrophy --fold "$FOLD" --mask-mode dk86 \
    --generated-dir "$GEN_DIR" --out-tag valeval --split val

echo "── Publishing to FAC ──"
bash "$HOME/taugennet/scripts/publish_to_fac.sh"
