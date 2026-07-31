#!/bin/bash
#SBATCH --job-name=condc_maskfix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=results/logs/condc_maskfix_%j.out
#SBATCH --error=results/logs/condc_maskfix_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# cond_c = spatial atrophy map + ptau217 cross-attention, retrained under the FIXED DK86
# mask, LEGACY fold-0 split (comparable to cond_a/cond_b). Train -> generate -> evaluate.
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
FOLD=0
AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/cond_compare_maskfix/spatialmap_ptau/fold_${FOLD}"
FIGTRAIN="results/figures/cond_compare_maskfix/spatialmap_ptau/fold_${FOLD}"
GEN="results/generated/dk86_regen/cond_c_ptau/fold_${FOLD}"
REC="results/records/dk86_glass/cond_c_ptau"
mkdir -p "$OUTDIR" "$FIGTRAIN" "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  fold ${FOLD}  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== Step 1: retrain (spatial map + ptau217), fixed mask, LEGACY split ==="
$PY scripts/train_spatial_atrophy.py \
    --skip-ae --ae-checkpoint "${AE_CKPT}" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --cond-mode ptau217 \
    --fold $FOLD --n-folds 5 \
    --diff-epochs 1200 --patience 200 --val-every 5 \
    --checkpoint-dir "${OUTDIR}" --figures-dir "${FIGTRAIN}" \
    --use-mask --no-use-mentor-split || { echo "TRAIN FAILED"; exit 1; }

CKPT="${OUTDIR}/diff_spatial_atrophy_best.pt"

echo "=== Step 2: generate (legacy fold-${FOLD}, cond=ptau217) ==="
$PY scripts/generate_spatial.py \
    --checkpoint "$CKPT" --generated-dir "$GEN" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --fold $FOLD --n-folds 5 --use-mask --no-use-mentor-split \
    || { echo "GEN FAILED"; exit 1; }

echo "=== Step 3: evaluate (DK86-masked, legacy fold-${FOLD}) + glass brain ==="
$PY scripts/evaluate_final.py \
    --mode atrophy --dataset spatial --cond-mode ptau217 \
    --fold $FOLD --n-folds 5 \
    --use-cached --generated-dir "$GEN" \
    --use-mask --no-use-mentor-split --skip-ablation --glass-brain \
    --figures-dir dk86_glass/cond_c_ptau \
    --records-dir "$REC" --metrics-out "$REC/metrics.json"

rm -f "${OUTDIR}/diff_spatial_atrophy.pt"
echo "Done: $(date)"
