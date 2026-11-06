#!/bin/bash
#SBATCH --job-name=eval_ab_off_prev2
#SBATCH --array=0-1
#SBATCH --partition=gpu
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/eval_ab_off_prev2_%A_%a.out
#SBATCH --error=results/logs/eval_ab_off_prev2_%A_%a.err

# PREVIEW (v2) eval of already-trained ab_emasnr_off folds 0-1, on a free g2 MIG
# (3g.20gb) slice via the high-priority gpu-test QOS so the backfill scheduler
# actually reaches it (NOT the A100 — inference fits easily in 20GB). Writes to
# separate *_preview2 dirs; touches no queued job (3288407 left pending as-is).

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
FOLD=$SLURM_ARRAY_TASK_ID
REC="results/records/eval/ab_emasnr_off_preview2/fold_${FOLD}"
GEN="results/generated/ab_emasnr_off_preview2/fold_${FOLD}"
mkdir -p "$REC" "$GEN" results/logs
echo "Job $SLURM_JOB_ID  fold ${FOLD}/5 (PREVIEW2/gpu-test)  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON scripts/evaluate_final.py \
    --mode atrophy --dataset final \
    --fold $FOLD --n-folds 5 \
    --checkpoint-dir results/checkpoints/ablation_emaminsnr/off/fold_${FOLD}/atrophy \
    --use-best \
    --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --use-mask --skip-ablation \
    --generated-dir "$GEN" \
    --figures-dir eval/ab_emasnr_off_preview2/fold_${FOLD} \
    --records-dir "$REC" \
    --metrics-out "$REC/metrics.json"

echo "Done: $(date)"
