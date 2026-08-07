#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/paint_%x_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/paint_%x_%j.err
PAINT=$1; TAG=$2
REPO=/home/remote/adebnath1/tau-denseunet-baseline
cd "$REPO"
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:$REPO:$TAUGENNET_ROOT:$PYTHONPATH"
export TAUGENNET_PAINT=$PAINT
PY=/usr/bin/python3.9

echo "=== TRAIN paint=$PAINT tag=$TAG $(date) ==="
$PY scripts/train.py --mode atrophy --mask-mode dk86 --lr 1e-4 --weight-decay 0.0 \
    --epochs 126 --patience 126 --val-frac 0.0 --use-controls-322 --use-suvr \
    --out-tag $TAG
if [ $? -ne 0 ]; then echo "TRAIN FAILED"; exit 1; fi

echo "=== GENERATE (held-out test) $(date) ==="
$PY scripts/generate.py --mode atrophy --use-controls-322 --split test --use-suvr \
    --checkpoint "$REPO/results/checkpoints/atrophy/$TAG/denseunet_best.pt" \
    --out-dir "$REPO/results/generated/$TAG"
if [ $? -ne 0 ]; then echo "GEN FAILED"; exit 1; fi

echo "=== EVALUATE $(date) ==="
$PY scripts/evaluate.py --mode atrophy \
    --generated-dir "$REPO/results/generated/$TAG" --out-tag $TAG --use-controls-322 \
    --records-dir "$REPO/results/records/denseunet_baseline/$TAG"
echo "=== DONE $TAG $(date) ==="
