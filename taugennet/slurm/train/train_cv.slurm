#!/bin/bash
#SBATCH --job-name=taugennet_cv
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/taugennet_cv_%A_%a.out
#SBATCH --error=results/logs/taugennet_cv_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# 5-fold cross-validation — one array task per fold.
# All 5 folds run in parallel; each trains AE + atrophy + ptau217 + combined.
# Submit: sbatch slurm/train/train_cv.slurm
# Or via orchestrator: python scripts/run_cv.py

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
FOLD=$SLURM_ARRAY_TASK_ID
N_FOLDS=5
AE_CKPT=results/checkpoints/cv/fold_${FOLD}/taugennet_checkpoint.pt

cd /scratch/network/sz3962/taugennet

mkdir -p results/logs \
         results/checkpoints/cv/fold_${FOLD}/atrophy \
         results/checkpoints/cv/fold_${FOLD}/ptau217 \
         results/checkpoints/cv/fold_${FOLD}/combined

echo "=== CV Fold ${FOLD} / ${N_FOLDS} ==="
echo "Job ID:     $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURM_NODELIST"
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Stage 1: AE pretraining (fold-specific to avoid test-set leakage) ─────────
echo "=== Stage 1: AE pretraining (fold ${FOLD}) ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --fold $FOLD --n-folds $N_FOLDS \
    --ae-epochs 100 \
    --diff-epochs 0 \
    --ae-checkpoint $AE_CKPT
echo "Stage 1 done at: $(date)"
echo ""

# ── Stage 2: Diffusion — atrophy ──────────────────────────────────────────────
echo "=== Stage 2: Diffusion (atrophy, fold ${FOLD}) ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --fold $FOLD --n-folds $N_FOLDS \
    --skip-ae --ae-checkpoint $AE_CKPT \
    --checkpoint-dir results/checkpoints/cv/fold_${FOLD}/atrophy \
    --diff-epochs 2000 --patience 50 --val-every 1 --monitor-every 25
echo "Stage 2 done at: $(date)"
echo ""

# ── Stage 3: Diffusion — ptau217 ──────────────────────────────────────────────
echo "=== Stage 3: Diffusion (ptau217, fold ${FOLD}) ==="
$PYTHON scripts/train.py \
    --mode ptau217 \
    --fold $FOLD --n-folds $N_FOLDS \
    --skip-ae --ae-checkpoint $AE_CKPT \
    --checkpoint-dir results/checkpoints/cv/fold_${FOLD}/ptau217 \
    --diff-epochs 2000 --patience 50 --val-every 1 --monitor-every 25
echo "Stage 3 done at: $(date)"
echo ""

# ── Stage 4: Diffusion — combined ─────────────────────────────────────────────
echo "=== Stage 4: Diffusion (combined, fold ${FOLD}) ==="
$PYTHON scripts/train.py \
    --mode combined \
    --fold $FOLD --n-folds $N_FOLDS \
    --skip-ae --ae-checkpoint $AE_CKPT \
    --checkpoint-dir results/checkpoints/cv/fold_${FOLD}/combined \
    --diff-epochs 2000 --patience 50 --val-every 1 --monitor-every 25
echo "Stage 4 done at: $(date)"
echo ""

echo "Fold ${FOLD} complete. End time: $(date)"
