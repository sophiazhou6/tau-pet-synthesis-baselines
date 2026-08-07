#!/bin/bash
#SBATCH --job-name=du_c322_geneval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/du_c322_geneval_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/du_c322_geneval_%j.err

set -uo pipefail
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
cd "$REPO"
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:$REPO:$TAUGENNET_ROOT:${PYTHONPATH:-}"

echo "=== STEP 1: generate cached predictions (controls_322, TEST split) — $(date) ==="
$PYTHON scripts/generate.py --mode atrophy \
    --use-controls-322 \
    --split test \
    --checkpoint "$REPO/results/checkpoints/atrophy/controls322_plain_lr1e-4_wd0_v2/denseunet_best.pt" \
    --out-dir "$REPO/results/generated/controls322_plain_v2"
GEN_RC=$?
echo "generate.py exit=$GEN_RC   $(date)"
if [ $GEN_RC -ne 0 ]; then echo "GENERATION FAILED — not scoring."; exit $GEN_RC; fi

echo ""
echo "=== STEP 2: score with canonical evaluate_final.py metrics — $(date) ==="
$PYTHON scripts/evaluate.py --mode atrophy \
    --generated-dir "$REPO/results/generated/controls322_plain_v2" \
    --out-tag controls322_plain_v2 \
    --use-controls-322 \
    --records-dir "$REPO/results/records/denseunet_baseline/controls322_plain_v2"
echo "evaluate.py exit=$?   $(date)"
echo "=== DONE: $(date) ==="
