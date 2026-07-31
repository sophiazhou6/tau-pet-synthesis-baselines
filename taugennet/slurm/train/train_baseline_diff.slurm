#!/bin/bash
#SBATCH --job-name=baseline_diff
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/baseline_diff_%j.out
#SBATCH --error=results/logs/baseline_diff_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/baseline_atrophy_small"
FIGDIR="results/figures/baseline_atrophy_small"
mkdir -p "$OUTDIR" "$FIGDIR" results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON scripts/train.py \
    --mode atrophy \
    --skip-ae \
    --ae-checkpoint "${AE_CKPT}" \
    --latent-ch 8 \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --diff-epochs 2000 \
    --patience 500 \
    --checkpoint-dir "${OUTDIR}" \
    --figures-dir "${FIGDIR}" \
    --use-mask

echo "Done: $(date)"
