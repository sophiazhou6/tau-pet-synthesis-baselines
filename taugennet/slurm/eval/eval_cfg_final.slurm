#!/bin/bash
#SBATCH --job-name=eval_cfg_final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_cfg_final_%j.out
#SBATCH --error=results/logs/eval_cfg_final_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
mkdir -p results/logs results/figures/cfg/atrophy results/figures/cfg/combined results/records/cfg

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── CFG atrophy ───────────────────────────────────────────────────────────────
echo "=== evaluate_final: cfg/atrophy ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/cfg/atrophy \
    --figures-dir cfg/atrophy \
    --records-dir results/records/cfg \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --use-best \
    --skip-ablation
echo "Atrophy done at: $(date)"
echo ""

# ── CFG combined ──────────────────────────────────────────────────────────────
echo "=== evaluate_final: cfg/combined ==="
$PYTHON scripts/evaluate_final.py \
    --mode combined \
    --checkpoint-dir results/checkpoints/cfg/combined \
    --figures-dir cfg/combined \
    --records-dir results/records/cfg \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --use-best \
    --skip-ablation
echo "Combined done at: $(date)"

echo "All done. End: $(date)"
