#!/bin/bash
#SBATCH --job-name=du_c322_suvr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/du_c322_suvr_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/du_c322_suvr_%j.err

REPO=/home/remote/adebnath1/tau-denseunet-baseline
cd "$REPO"
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:$REPO:$TAUGENNET_ROOT:$PYTHONPATH"
PY=/usr/bin/python3.9

echo "=== painted SUVR (--use-suvr) — $(date) ==="
$PY scripts/train.py --mode atrophy --mask-mode dk86 --lr 1e-4 --weight-decay 0.0 \
    --epochs 126 --patience 126 --val-frac 0.0 --use-controls-322 --use-suvr \
    --out-tag controls322_suvr_lr1e-4_wd0
echo "painted exit=$?  $(date)"

echo "=== FiLM SUVR (--use-suvr-film) — $(date) ==="
$PY scripts/train.py --mode atrophy --mask-mode dk86 --lr 1e-4 --weight-decay 0.0 \
    --epochs 126 --patience 126 --val-frac 0.0 --use-controls-322 --use-suvr-film \
    --out-tag controls322_suvrfilm_lr1e-4_wd0
echo "film exit=$?  $(date)"
echo "=== DONE $(date) ==="
