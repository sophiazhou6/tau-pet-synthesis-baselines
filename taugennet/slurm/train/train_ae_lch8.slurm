#!/bin/bash
#SBATCH --job-name=ae_lch8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/ae_lch8_%j.out
#SBATCH --error=results/logs/ae_lch8_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

OUTDIR="results/checkpoints/ae_lch8_dk86"
FIGDIR="results/figures/ae_lch8_dk86"
mkdir -p "$OUTDIR" "$FIGDIR" results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON scripts/train_lp_modality.py \
    --mode atrophy \
    --diff-epochs 0 \
    --ae-epochs 400 \
    --patience 50 \
    --latent-ch 8 \
    --lp-weight 0.0 \
    --modality-weight 0.0 \
    --use-mask \
    --ae-checkpoint "${OUTDIR}/ae_checkpoint.pt" \
    --checkpoint-dir "${OUTDIR}" \
    --figures-dir "${FIGDIR}"

echo "Done: $(date)"
