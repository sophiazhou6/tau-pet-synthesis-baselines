#!/bin/bash
#SBATCH --job-name=denseunet_phase6
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/phase6_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/phase6_%j.err
set -euo pipefail
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
cd "$REPO"
source "$HOME/taugennet/scripts/wait_for_fac.sh"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$REPO:$HOME/taugennet:${PYTHONPATH:-}"
mkdir -p results/logs

$PYTHON scripts/train.py --mode atrophy --mask-mode dk86 \
    --lr 1e-4 --weight-decay 0.0 --epochs 126 --patience 126 \
    --use-phase6-controls --phase6-fold 0 --out-tag phase6_controls --overwrite
