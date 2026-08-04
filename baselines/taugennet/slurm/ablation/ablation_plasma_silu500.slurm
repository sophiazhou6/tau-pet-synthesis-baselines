#!/bin/bash
#SBATCH --job-name=abl_plasma
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/logs/ablation_plasma_%j.out
#SBATCH --error=results/logs/ablation_plasma_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Conditioning ablation for the silu_500ep COMBINED model (plasma-shortcut test).
# Regenerates the test set under full / no_plasma / no_atrophy, same checkpoint +
# DDPM-500 pipeline as baseline. Writes ONLY to a new combined_ablation/ subtree.

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
CKPT=results/checkpoints/silu/combined_500ep/diff_combined_best.pt
OUT_BASE=results/generated/silu_500ep/combined_ablation

cd /scratch/network/sz3962/taugennet
mkdir -p results/logs "$OUT_BASE"

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

$PYTHON scripts/ablation_plasma.py \
    --mode combined \
    --arch silu \
    --checkpoint "$CKPT" \
    --out-base "$OUT_BASE" \
    --conditions full no_plasma no_atrophy \
    --n-steps 500

echo ""
echo "Done. End time: $(date)"
echo "Score with, e.g.:"
echo "  $PYTHON scripts/crosssubject_2x2.py --mode combined --arch silu --generated-dir $OUT_BASE/no_plasma"
