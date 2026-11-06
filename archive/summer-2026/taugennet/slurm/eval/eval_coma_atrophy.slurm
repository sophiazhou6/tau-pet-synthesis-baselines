#!/bin/bash
#SBATCH --job-name=eval_coma_atrophy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=results/logs/eval_coma_atrophy_%j.out
#SBATCH --error=results/logs/eval_coma_atrophy_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

CKPT_DIR="results/checkpoints/coma_atrophy_small"
FIG_DIR="results/figures/coma_atrophy_small/eval"
REC_DIR="results/records/coma_atrophy_small"
GEN_DIR="results/generated/coma_atrophy_small/atrophy"

# Find the best checkpoint file
CKPT="${CKPT_DIR}/diff_dynamic_prompt_best.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="${CKPT_DIR}/diff_dynamic_prompt.pt"
fi

mkdir -p "$FIG_DIR" "$REC_DIR" "$GEN_DIR" results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Step 1: Run DDPM to generate tau PET for each test subject
echo "=== Step 1: Generating CoMA model outputs — $(date) ==="
$PYTHON scripts/generate_coma.py \
    --checkpoint "${CKPT}" \
    --generated-dir "${GEN_DIR}" \
    --latent-ch 8 \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --n-steps 500 \
    --use-mask
echo "Generation done: $(date)"

# Step 2: Compute metrics and figures from cached generations
echo "=== Step 2: Computing metrics from cached outputs — $(date) ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy \
    --use-cached \
    --generated-dir "${GEN_DIR}" \
    --figures-dir coma_atrophy_small/eval \
    --records-dir "${REC_DIR}" \
    --arch silu \
    --use-mask \
    --skip-ablation

echo "Done: $(date)"
