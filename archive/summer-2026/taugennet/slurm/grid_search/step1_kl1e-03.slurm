#!/bin/bash
#SBATCH --job-name=gs_s1_kl1e-03
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/gs_s1_kl1e-03_%A_%a.out
#SBATCH --error=results/logs/gs_s1_kl1e-03_%A_%a.err

# Grid search Step 1: kl_weight=0.001, lr=0.0001 (fixed)
# Full AE + atrophy diffusion pipeline, one array task per fold.
# Do NOT sbatch this directly — use: python scripts/run_grid_search.py --step 1
# Then submit manually: sbatch slurm/grid_search/step1_kl1e-03.slurm

PYTHON=/usr/bin/python3.9
FOLD=$SLURM_ARRAY_TASK_ID
N_FOLDS=5
AE_CKPT=results/checkpoints/grid_search/step1/kl1e-03/fold_${FOLD}/taugennet_checkpoint.pt

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/logs \
         results/checkpoints/grid_search/step1/kl1e-03/fold_${FOLD}/atrophy

echo "=== Grid Search Step 1: kl_weight=0.001, fold ${FOLD}/${N_FOLDS} ==="
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
    --kl-weight 0.001
echo "AE done at: $(date)"
echo ""

# ── Diffusion training ──────────────────────────────────────────────────
echo "=== Diffusion training ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --fold $FOLD --n-folds $N_FOLDS \
    --skip-ae --ae-checkpoint $AE_CKPT \
    --checkpoint-dir results/checkpoints/grid_search/step1/kl1e-03/fold_${FOLD}/atrophy \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 50 \
    --lr 0.0001 \
    --kl-weight 0.001
echo "Diffusion done at: $(date)"

echo "Fold ${FOLD} complete. End: $(date)"
