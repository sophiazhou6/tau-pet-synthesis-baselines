#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/paintcv_%x_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/paintcv_%x_%j.err
PAINT=$1; LR=$2; WD=$3; TAG=$4; FOLD=$5
REPO=/home/remote/adebnath1/tau-denseunet-baseline
cd "$REPO"
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:$REPO:$TAUGENNET_ROOT:$PYTHONPATH"
export TAUGENNET_PAINT=$PAINT
echo "=== paintcv paint=$PAINT lr=$LR wd=$WD fold=$FOLD $(date) ==="
/usr/bin/python3.9 scripts/train.py --mode atrophy --mask-mode dk86 \
    --lr $LR --weight-decay $WD --epochs 126 --patience 126 \
    --use-suvr --use-controls-322 --fold $FOLD --n-folds 5 --out-tag $TAG
echo "exit=$? $(date)"
