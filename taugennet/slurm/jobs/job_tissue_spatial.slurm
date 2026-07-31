#!/bin/bash
#SBATCH --job-name=spatial_tissue
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/spatial_tissue_%j.out
#SBATCH --error=results/logs/spatial_tissue_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Spatial atrophy + GM/WM/CSF tissue conditioning (UNet in_ch 7 -> 10).
# Isolated checkpoint/figure dirs so this never overwrites the atrophy-only run.

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/spatial_atrophy_tissue"
FIGDIR="results/figures/spatial_atrophy_tissue"
mkdir -p "$OUTDIR" "$FIGDIR" results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON scripts/train_spatial_atrophy.py \
    --skip-ae \
    --ae-checkpoint "${AE_CKPT}" \
    --latent-ch 8 \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --diff-epochs 2000 \
    --patience 500 \
    --checkpoint-dir "${OUTDIR}" \
    --figures-dir "${FIGDIR}" \
    --use-tissue \
    --use-mask

echo "Done: $(date)"
