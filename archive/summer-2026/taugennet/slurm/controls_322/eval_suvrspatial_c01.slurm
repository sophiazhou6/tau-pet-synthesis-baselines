#!/bin/bash
#SBATCH --job-name=eval_suvrspatial_c322
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_suvrspatial_c322_%j.out
#SBATCH --error=results/logs/eval_suvrspatial_c322_%j.err

set -uo pipefail
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
export TAUGENNET_SPATIAL_MAP=suvr
PY=/usr/bin/python3.9

CKPT=$(ls -t results/checkpoints/controls_322/suvrspatial_c01/*best*.pt 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls -t results/checkpoints/controls_322/suvrspatial_c01/*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "NO CHECKPOINT FOUND — aborting"; exit 1; fi
echo "Using checkpoint: $CKPT"

echo "=== STEP 1: generate (spatial SUVR-map, controls_322, VAL) — $(date) ==="
$PY scripts/generate_spatial.py \
    --checkpoint "$CKPT" \
    --generated-dir results/generated/controls_322_final/suvrspatial_c01 \
    --use-controls-322 --eval-on val \
    --latent-ch 2 --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --use-mask
RC=$?; echo "generate exit=$RC  $(date)"
if [ $RC -ne 0 ]; then echo "GENERATION FAILED — not scoring."; exit $RC; fi

echo ""
echo "=== STEP 2: canonical SET1/SET2/SET3 + glass brain — $(date) ==="
$PY scripts/evaluate_final.py \
    --mode atrophy --dataset spatial --cond-mode none \
    --use-controls-322 --eval-on val --use-cached \
    --generated-dir results/generated/controls_322_final/suvrspatial_c01 \
    --records-dir results/records/controls_322_final/suvrspatial_c01 \
    --figures-dir silu/controls_322_final/suvrspatial_c01 \
    --use-mask --glass-brain --skip-ablation
echo "evaluate exit=$?  $(date)"
echo "=== DONE: $(date) ==="
