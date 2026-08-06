#!/bin/bash
#SBATCH --job-name=eval_cond_c
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_cond_c_%A_%a.out
#SBATCH --error=results/logs/eval_cond_c_%A_%a.err

# Full eval for variant (c) spatial map + ptau217 CLIP — ALL 5 folds (CV).
# Spatial model: generate_spatial then evaluate_final --use-cached. N~37 test/fold.

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
FOLD=$SLURM_ARRAY_TASK_ID
CKPT="results/checkpoints/cond_compare/spatialmap_ptau/fold_${FOLD}/diff_spatial_atrophy_best.pt"
GEN="results/generated/cond_compare/spatialmap_ptau/fold_${FOLD}"
REC="results/records/eval/cond_c_ptau/fold_${FOLD}"
mkdir -p "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  fold ${FOLD}/5  Node: $SLURM_NODELIST  Start: $(date)"

echo "=== Step 1: generate (spatial, fold ${FOLD}, cond=ptau217) ==="
$PYTHON scripts/generate_spatial.py \
    --checkpoint "$CKPT" --generated-dir "$GEN" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --fold $FOLD --n-folds 5 --use-mask

echo "=== Step 2: metrics + figures (DK86-masked, same fold-${FOLD} test) ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy --dataset spatial --cond-mode ptau217 \
    --fold $FOLD --n-folds 5 \
    --use-cached --generated-dir "$GEN" \
    --use-mask --skip-ablation \
    --figures-dir eval/cond_c_ptau/fold_${FOLD} \
    --records-dir "$REC" \
    --metrics-out "$REC/metrics.json"

echo "Done: $(date)"
