#!/bin/bash
#SBATCH --job-name=eval_cond_a
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=results/logs/eval_cond_a_%A_%a.out
#SBATCH --error=results/logs/eval_cond_a_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Full evaluate_final for conditioning variant (a) atrophy MLP — ALL 5 folds (CV).

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
FOLD=$SLURM_ARRAY_TASK_ID
REC="results/records/eval/cond_a_mlp/fold_${FOLD}"
GEN="results/generated/cond_a_mlp/fold_${FOLD}"
mkdir -p "$REC" "$GEN" results/logs
echo "Job $SLURM_JOB_ID  fold ${FOLD}/5  Node: $SLURM_NODELIST  Start: $(date)"

$PYTHON scripts/evaluate_final.py \
    --mode atrophy --dataset final \
    --fold $FOLD --n-folds 5 \
    --checkpoint-dir results/checkpoints/cond_compare/mlp_atrophy/fold_${FOLD}/atrophy \
    --use-best \
    --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --use-mask --skip-ablation \
    --generated-dir "$GEN" \
    --figures-dir eval/cond_a_mlp/fold_${FOLD} \
    --records-dir "$REC" \
    --metrics-out "$REC/metrics.json"

echo "Done: $(date)"
