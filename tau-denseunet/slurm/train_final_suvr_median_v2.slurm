#!/bin/bash
#SBATCH --job-name=final_suvr_median_lr1e-4_wd0
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/final_suvr_median_lr1e-4_wd0_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/final_suvr_median_lr1e-4_wd0_%j.err
set -euo pipefail
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
cd "$REPO"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$REPO:$HOME/taugennet:${PYTHONPATH:-}"
mkdir -p results/logs

$PYTHON scripts/train.py --mode atrophy --mask-mode dk86 \
    --lr 1e-4 --weight-decay 0.0 --epochs 49 --patience 49 \
    --val-frac 0.0 --use-suvr --out-tag final_suvr_median_lr1e-4_wd0 --overwrite \
    --mentor-split-dir /home/remote/adebnath1/taugennet/data/raw/legacy_admci_split
