#!/bin/bash
#SBATCH --job-name=eval_ab_off_prev
#SBATCH --array=0-1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=results/logs/eval_ab_off_prev_%A_%a.out
#SBATCH --error=results/logs/eval_ab_off_prev_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# PREVIEW eval of the already-trained ab_emasnr_off folds 0-1, run NOW on a free
# g2 MIG (3g.20gb) slice to dodge the jammed A100 queue. Inference-only, ~20GB is
# plenty. Writes to SEPARATE *_preview dirs so it does NOT touch or overwrite the
# queued canonical eval_ab_off (3286770) — that job still runs later as planned.

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
FOLD=$SLURM_ARRAY_TASK_ID
REC="results/records/eval/ab_emasnr_off_preview/fold_${FOLD}"
GEN="results/generated/ab_emasnr_off_preview/fold_${FOLD}"
mkdir -p "$REC" "$GEN" results/logs
echo "Job $SLURM_JOB_ID  fold ${FOLD}/5 (PREVIEW)  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON scripts/evaluate_final.py \
    --mode atrophy --dataset final \
    --fold $FOLD --n-folds 5 \
    --checkpoint-dir results/checkpoints/ablation_emaminsnr/off/fold_${FOLD}/atrophy \
    --use-best \
    --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --use-mask --skip-ablation \
    --generated-dir "$GEN" \
    --figures-dir eval/ab_emasnr_off_preview/fold_${FOLD} \
    --records-dir "$REC" \
    --metrics-out "$REC/metrics.json"

echo "Done: $(date)"
