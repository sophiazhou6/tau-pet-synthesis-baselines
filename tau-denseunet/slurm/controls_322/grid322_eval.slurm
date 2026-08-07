#!/bin/bash
#SBATCH --job-name=grid322_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/grid322_eval_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/grid322_eval_%j.err
set -uo pipefail
REPO=/home/remote/adebnath1/tau-denseunet-baseline
cd "$REPO"
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:$REPO:$TAUGENNET_ROOT:$PYTHONPATH"
unset TAUGENNET_PAINT
PY=/usr/bin/python3.9
for LR in 5e-5 1e-4 3e-4; do for WD in 0.0 1e-4; do
  TAG=grid322_lr${LR}_wd${WD}
  for F in 0 1 2 3 4; do
    CK="$REPO/results/checkpoints/atrophy/$TAG/fold_$F/denseunet_best.pt"
    if [ ! -f "$CK" ]; then echo "SKIP $TAG f$F"; continue; fi
    GD="$REPO/results/generated/${TAG}_valeval/fold_$F"
    RD="$REPO/results/records/denseunet_baseline/cv322_${TAG}/fold_$F"
    mkdir -p "$GD" "$RD"
    echo "=== $TAG fold=$F $(date) ==="
    $PY scripts/generate.py --mode atrophy --use-controls-322 --split val \
        --fold $F --n-folds 5 --checkpoint "$CK" --out-dir "$GD"
    if [ $? -ne 0 ]; then echo "GEN FAILED $TAG f$F"; continue; fi
    $PY scripts/evaluate.py --mode atrophy --fold $F --n-folds 5 \
        --generated-dir "$GD" --out-tag "cv322_${TAG}" --use-controls-322 \
        --eval-on val --records-dir "$RD" --no-glass-brain
  done
done; done
echo "=== GRID322 EVAL DONE $(date) ==="
