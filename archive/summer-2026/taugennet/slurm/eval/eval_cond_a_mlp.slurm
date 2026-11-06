#!/bin/bash
#SBATCH --job-name=eval_cond_a
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=results/logs/eval_cond_a_%A_%a.out
#SBATCH --error=results/logs/eval_cond_a_%A_%a.err

# Full evaluate_final for conditioning variant (a) atrophy MLP — ALL 5 folds (CV).

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
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
