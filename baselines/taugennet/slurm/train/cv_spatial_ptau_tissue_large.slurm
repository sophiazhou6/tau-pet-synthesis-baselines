#!/bin/bash
#SBATCH --job-name=cv_sptt_lg
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/cv_sptt_lg_%A_%a.out
#SBATCH --error=results/logs/cv_sptt_lg_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# 5-fold CV (MENTOR split) — spatial atrophy map + ptau217 + tissue one-hot,
# LARGER UNet (256,512,768, n_transformer=3). train -> generate -> evaluate per fold.
# Everything (train/generate/eval) uses the mentor split (default), so no leakage and
# apples-to-apples with Anil. Generations always cached; eval on DK86 with correct SUVR.
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
FOLD=$SLURM_ARRAY_TASK_ID
AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/cv_spatial_ptau_tissue_large/fold_${FOLD}"
FIGTRAIN="results/figures/cv_spatial_ptau_tissue_large/fold_${FOLD}"
GEN="results/generated/cv_spatial_ptau_tissue_large/fold_${FOLD}"
REC="results/records/cv_spatial_ptau_tissue_large/fold_${FOLD}"
mkdir -p "$OUTDIR" "$FIGTRAIN" "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  fold ${FOLD}/5  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== Step 1: train (spatial map + ptau + tissue, LARGE arch, mentor split fold ${FOLD}) ==="
$PY scripts/train_spatial_atrophy.py \
    --skip-ae --ae-checkpoint "${AE_CKPT}" \
    --latent-ch 8 --unet-channels 256,512,768 --n-transformer 3 \
    --cond-mode ptau217 --use-tissue \
    --fold $FOLD --n-folds 5 --use-mentor-split \
    --diff-epochs 1200 --patience 200 --val-every 5 \
    --checkpoint-dir "${OUTDIR}" --figures-dir "${FIGTRAIN}" \
    --use-mask || { echo "TRAIN FAILED"; exit 1; }

CKPT="${OUTDIR}/diff_spatial_atrophy_best.pt"

echo "=== Step 2: generate (mentor split fold ${FOLD}) — cached ==="
$PY scripts/generate_spatial.py \
    --checkpoint "$CKPT" --generated-dir "$GEN" \
    --latent-ch 8 --unet-channels 256,512,768 --n-transformer 3 \
    --n-steps 500 --fold $FOLD --n-folds 5 --use-mask --use-mentor-split \
    || { echo "GEN FAILED"; exit 1; }

echo "=== Step 3: evaluate (DK86, correct SUVR, mentor split) + glass brain ==="
$PY scripts/evaluate_final.py \
    --mode atrophy --dataset spatial --cond-mode ptau217 \
    --fold $FOLD --n-folds 5 \
    --use-cached --generated-dir "$GEN" \
    --use-mask --use-mentor-split --skip-ablation --glass-brain \
    --figures-dir cv_spatial_ptau_tissue_large/fold_${FOLD} \
    --records-dir "$REC" --metrics-out "$REC/metrics.json"

rm -f "${OUTDIR}/diff_spatial_atrophy.pt"
echo "Done fold ${FOLD}: $(date)"
