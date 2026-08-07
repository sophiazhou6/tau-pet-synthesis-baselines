#!/bin/bash
#SBATCH --job-name=denseunet_suvr
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/final_suvr_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/final_suvr_%A_%a.err
set -euo pipefail
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
cd "$REPO"
source "$HOME/taugennet/scripts/wait_for_fac.sh"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$REPO:$HOME/taugennet:${PYTHONPATH:-}"
mkdir -p results/logs

$PYTHON scripts/train.py --mode atrophy --mask-mode dk86 --fold "$FOLD" --n-folds 5 \
    --lr 1e-4 --weight-decay 0.0 --epochs 300 --patience 15 \
    --use-suvr --out-tag final_suvr_lr1e-4_wd0
