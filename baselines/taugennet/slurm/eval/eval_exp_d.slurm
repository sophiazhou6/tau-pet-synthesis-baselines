#!/bin/bash
#SBATCH --job-name=eval_exp_d
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=results/logs/eval_exp_d_%A_%a.out
#SBATCH --error=results/logs/eval_exp_d_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
FOLD=$SLURM_ARRAY_TASK_ID
CKPT="results/checkpoints/lr_ablation/exp_d/fold_${FOLD}/diff_spatial_atrophy_best.pt"
GEN="results/generated/lr_ablation/exp_d/fold_${FOLD}"
REC="results/records/eval/lr_ablation/exp_d/fold_${FOLD}"
mkdir -p "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  fold ${FOLD}/5  Node: $SLURM_NODELIST  Start: $(date)"

echo "=== Step 1: generate (exp_d, fold ${FOLD}) ==="
# exp_d trains with --noise-schedule cosine; the sampler MUST use the same schedule.
$PYTHON scripts/generate_spatial.py \
    --checkpoint "$CKPT" --generated-dir "$GEN" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --fold $FOLD --n-folds 5 --use-mask \
    --noise-schedule cosine

echo "=== Step 2: metrics + figures ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy --dataset spatial --cond-mode none \
    --fold $FOLD --n-folds 5 \
    --use-cached --generated-dir "$GEN" \
    --use-mask --skip-ablation \
    --figures-dir eval/lr_ablation/exp_d/fold_${FOLD} \
    --records-dir "$REC" \
    --metrics-out "$REC/metrics.json"

echo "Done: $(date)"
