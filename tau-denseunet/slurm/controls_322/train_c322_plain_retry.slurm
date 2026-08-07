#!/bin/bash
#SBATCH --job-name=denseunet_c322_plain_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/c322_plain_v2_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/c322_plain_v2_%j.err
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
cd "$REPO"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$REPO:$HOME/taugennet:${PYTHONPATH:-}"
mkdir -p results/logs
$PYTHON scripts/train.py --mode atrophy --mask-mode dk86 \
    --lr 1e-4 --weight-decay 0.0 --epochs 126 --patience 126 \
    --val-frac 0.0 --use-controls-322 --out-tag controls322_plain_lr1e-4_wd0_v2
