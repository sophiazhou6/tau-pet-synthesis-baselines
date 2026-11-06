#!/bin/bash
#SBATCH --job-name=ae_ablation
#SBATCH --output=results/logs/ae_ablation_%A_%a.out
#SBATCH --error=results/logs/ae_ablation_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/logs results/checkpoints/ae_exp_masked/{baseline,lp,modality,both} \
         results/figures/ae_exp_masked/{baseline,lp,modality,both}

echo "Job ID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

case $SLURM_ARRAY_TASK_ID in
    0)
        echo "=== Condition: baseline (L1 + KL only) ==="
        $PYTHON scripts/train_lp_modality.py \
            --mode atrophy \
            --ae-epochs 400 \
            --patience 100 \
            --diff-epochs 0 \
            --ae-checkpoint results/checkpoints/ae_exp_masked/baseline/ae_checkpoint.pt \
            --checkpoint-dir results/checkpoints/ae_exp_masked/baseline \
            --figures-dir results/figures/ae_exp_masked/baseline
        ;;
    1)
        echo "=== Condition: LP loss (lp-weight=0.1) ==="
        $PYTHON scripts/train_lp_modality.py \
            --mode atrophy \
            --ae-epochs 400 \
            --patience 100 \
            --diff-epochs 0 \
            --lp-weight 0.1 \
            --ae-checkpoint results/checkpoints/ae_exp_masked/lp/ae_checkpoint.pt \
            --checkpoint-dir results/checkpoints/ae_exp_masked/lp \
            --figures-dir results/figures/ae_exp_masked/lp
        ;;
    2)
        echo "=== Condition: Modality loss (modality-weight=0.05) ==="
        $PYTHON scripts/train_lp_modality.py \
            --mode atrophy \
            --ae-epochs 400 \
            --patience 100 \
            --diff-epochs 0 \
            --modality-weight 0.05 \
            --ae-checkpoint results/checkpoints/ae_exp_masked/modality/ae_checkpoint.pt \
            --checkpoint-dir results/checkpoints/ae_exp_masked/modality \
            --figures-dir results/figures/ae_exp_masked/modality
        ;;
    3)
        echo "=== Condition: Both (lp-weight=0.1, modality-weight=0.05) ==="
        $PYTHON scripts/train_lp_modality.py \
            --mode atrophy \
            --ae-epochs 400 \
            --patience 100 \
            --diff-epochs 0 \
            --lp-weight 0.1 \
            --modality-weight 0.05 \
            --ae-checkpoint results/checkpoints/ae_exp_masked/both/ae_checkpoint.pt \
            --checkpoint-dir results/checkpoints/ae_exp_masked/both \
            --figures-dir results/figures/ae_exp_masked/both
        ;;
esac

echo "Done: $(date)"