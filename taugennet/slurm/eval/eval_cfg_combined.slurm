#!/bin/bash
#SBATCH --job-name=eval_cfg_combined
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=results/logs/eval_cfg_combined_%j.out
#SBATCH --error=results/logs/eval_cfg_combined_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
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
