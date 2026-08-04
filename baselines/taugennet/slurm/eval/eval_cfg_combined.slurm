#!/bin/bash
#SBATCH --job-name=eval_cfg_combined
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=results/logs/eval_cfg_combined_%j.out
#SBATCH --error=results/logs/eval_cfg_combined_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
mkdir -p results/logs results/figures/cfg/combined results/records/cfg results/generated/cfg/combined

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Evaluating CFG combined ==="
$PYTHON scripts/evaluate_cfg.py \
    --mode combined \
    --checkpoint-dir results/checkpoints/cfg/combined \
    --figures-dir results/figures/cfg/combined \
    --records-dir results/records/cfg \
    --generated-dir results/generated/cfg/combined \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --use-best \
    --skip-ablation

echo "Done at: $(date)"
