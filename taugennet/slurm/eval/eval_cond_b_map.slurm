#!/bin/bash
#SBATCH --job-name=eval_cond_b
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_cond_b_%A_%a.out
#SBATCH --error=results/logs/eval_cond_b_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Full eval for variant (b) spatial map only — ALL 5 folds (CV). Spatial model:
# generate_spatial (matching dataset/order) then evaluate_final --use-cached.

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
FOLD=$SLURM_ARRAY_TASK_ID
CKPT="results/checkpoints/cond_compare/spatialmap_none/fold_${FOLD}/diff_spatial_atrophy_best.pt"
GEN="results/generated/cond_compare/spatialmap_none/fold_${FOLD}"
REC="results/records/eval/cond_b_map/fold_${FOLD}"
mkdir -p "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  fold ${FOLD}/5  Node: $SLURM_NODELIST  Start: $(date)"

echo "=== Step 1: generate (spatial, fold ${FOLD}, cond=none) ==="
$PYTHON scripts/generate_spatial.py \
    --checkpoint "$CKPT" --generated-dir "$GEN" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --fold $FOLD --n-folds 5 --use-mask

echo "=== Step 2: metrics + figures (DK86-masked, same fold-${FOLD} test) ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy --dataset spatial --cond-mode none \
    --fold $FOLD --n-folds 5 \
    --use-cached --generated-dir "$GEN" \
    --use-mask --skip-ablation \
    --figures-dir eval/cond_b_map/fold_${FOLD} \
    --records-dir "$REC" \
    --metrics-out "$REC/metrics.json"

echo "Done: $(date)"
