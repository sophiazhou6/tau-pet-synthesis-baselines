#!/bin/bash
#SBATCH --job-name=eval_baseline_atrophy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_baseline_atrophy_%j.out
#SBATCH --error=results/logs/eval_baseline_atrophy_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

CKPT_DIR="results/checkpoints/baseline_atrophy_small"
FIG_DIR="results/figures/baseline_atrophy_small/eval"
REC_DIR="results/records/baseline_atrophy_small"
GEN_DIR="results/generated/baseline_atrophy_small/atrophy"

mkdir -p "$FIG_DIR" "$REC_DIR" "$GEN_DIR" results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON scripts/evaluate_final.py \
    --mode atrophy \
    --checkpoint-dir "${CKPT_DIR}" \
    --figures-dir baseline_atrophy_small/eval \
    --records-dir "${REC_DIR}" \
    --arch silu \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --use-mask \
    --n-steps 500 \
    --save-generated \
    --generated-dir "${GEN_DIR}" \
    --use-best

echo "Done: $(date)"
