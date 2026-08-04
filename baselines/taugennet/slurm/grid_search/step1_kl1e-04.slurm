#!/bin/bash
#SBATCH --job-name=gs_s1_kl1e-04
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/gs_s1_kl1e-04_%A_%a.out
#SBATCH --error=results/logs/gs_s1_kl1e-04_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Grid search Step 1: kl_weight=0.0001, lr=0.0001 (fixed), AE only
# One array task per fold.
# Do NOT sbatch this directly — use: python scripts/run_grid_search.py --step 1
# Then submit manually: sbatch slurm/grid_search/step1_kl1e-04.slurm

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
FOLD=$SLURM_ARRAY_TASK_ID
N_FOLDS=5
AE_CKPT=results/checkpoints/grid_search/step1/kl1e-04/fold_${FOLD}/taugennet_checkpoint.pt

cd /scratch/network/sz3962/taugennet

mkdir -p results/logs \
         results/checkpoints/grid_search/step1/kl1e-04/fold_${FOLD}/atrophy

echo "=== Grid Search Step 1: kl_weight=0.0001, fold ${FOLD}/${N_FOLDS} (AE only) ==="
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── AE pretraining ──────────────────────────────────────────────────────
echo "=== AE pretraining ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --fold $FOLD --n-folds $N_FOLDS \
    --ae-epochs 100 \
    --diff-epochs 0 \
    --ae-checkpoint $AE_CKPT \
    --kl-weight 0.0001
echo "AE done at: $(date)"
echo ""

echo "Fold ${FOLD} complete. End: $(date)"
