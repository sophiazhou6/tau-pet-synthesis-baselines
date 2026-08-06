#!/bin/bash
#SBATCH --job-name=eval_spatial_tissue
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=results/logs/eval_spatial_tissue_%j.out
#SBATCH --error=results/logs/eval_spatial_tissue_%j.err

# Evaluate the spatial atrophy + tissue model. generate_spatial.py reads
# use_tissue/extra_cond_ch from the checkpoint, so no extra flag is needed.

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

CKPT_DIR="results/checkpoints/spatial_atrophy_tissue"
FIG_DIR="results/figures/spatial_atrophy_tissue/eval"
REC_DIR="results/records/spatial_atrophy_tissue"
GEN_DIR="results/generated/spatial_atrophy_tissue/atrophy"

CKPT="${CKPT_DIR}/diff_spatial_atrophy_best.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="${CKPT_DIR}/diff_spatial_atrophy.pt"
fi

mkdir -p "$FIG_DIR" "$REC_DIR" "$GEN_DIR" results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Step 1: Generating spatial atrophy+tissue model outputs — $(date) ==="
$PYTHON scripts/generate_spatial.py \
    --checkpoint "${CKPT}" \
    --generated-dir "${GEN_DIR}" \
    --latent-ch 8 \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --n-steps 500 \
    --use-mask
echo "Generation done: $(date)"

echo "=== Step 2: Computing metrics from cached outputs — $(date) ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy \
    --use-cached \
    --generated-dir "${GEN_DIR}" \
    --figures-dir spatial_atrophy_tissue/eval \
    --records-dir "${REC_DIR}" \
    --arch silu \
    --use-mask \
    --skip-ablation

echo "Done: $(date)"
