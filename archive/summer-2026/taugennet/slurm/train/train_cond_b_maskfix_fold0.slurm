#!/bin/bash
#SBATCH --job-name=condb_maskfix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/condb_maskfix_%j.out
#SBATCH --error=results/logs/condb_maskfix_%j.err

# Validate the dataset_spatial mask fix: retrain cond_b (spatial-map only, cond=none)
# on the SHARED DK86 binary mask the AE was trained on, then generate + evaluate.
# fold 0 only — quick end-to-end check. Writes to *_maskfix paths (no overwrite).

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
FOLD=0
AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/cond_compare_maskfix/spatialmap_none/fold_${FOLD}"
FIGTRAIN="results/figures/cond_compare_maskfix/spatialmap_none/fold_${FOLD}"
GEN="results/generated/cond_compare_maskfix/spatialmap_none/fold_${FOLD}"
REC="results/records/eval/cond_b_maskfix/fold_${FOLD}"
mkdir -p "$OUTDIR" "$FIGTRAIN" "$GEN" "$REC" results/logs

echo "Job $SLURM_JOB_ID  fold ${FOLD}  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Step 1: retrain spatial diffusion (cond=none) on the fixed shared DK86 mask ==="
$PYTHON scripts/train_spatial_atrophy.py \
    --skip-ae --ae-checkpoint "${AE_CKPT}" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --cond-mode none \
    --fold $FOLD --n-folds 5 \
    --diff-epochs 1200 --patience 200 --val-every 5 \
    --checkpoint-dir "${OUTDIR}" --figures-dir "${FIGTRAIN}" \
    --use-mask || { echo "TRAIN FAILED"; exit 1; }

CKPT="${OUTDIR}/diff_spatial_atrophy_best.pt"

echo "=== Step 2: generate (spatial, fold ${FOLD}, cond=none, 500 steps) ==="
$PYTHON scripts/generate_spatial.py \
    --checkpoint "$CKPT" --generated-dir "$GEN" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --fold $FOLD --n-folds 5 --use-mask || { echo "GEN FAILED"; exit 1; }

echo "=== Step 3: evaluate (DK86-masked, same fold-${FOLD} test) ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy --dataset spatial --cond-mode none \
    --fold $FOLD --n-folds 5 \
    --use-cached --generated-dir "$GEN" \
    --use-mask --skip-ablation \
    --figures-dir eval/cond_b_maskfix/fold_${FOLD} \
    --records-dir "$REC" \
    --metrics-out "$REC/metrics.json"

rm -f "${OUTDIR}/diff_spatial_atrophy.pt"
echo "Done: $(date)"
echo "Metrics: $REC/metrics.json"
