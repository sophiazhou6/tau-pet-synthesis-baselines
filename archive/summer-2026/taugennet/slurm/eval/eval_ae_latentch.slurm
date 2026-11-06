#!/bin/bash
#SBATCH --job-name=eval_ae_lch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=results/logs/eval_ae_lch_%j.out
#SBATCH --error=results/logs/eval_ae_lch_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
mkdir -p results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TAGS=(lch4_lp0.0 lch4_lp0.1 lch4_lp0.5 lch4_lp1.0 lch8_lp0.0 lch8_lp0.1 lch8_lp0.5 lch8_lp1.0)

for TAG in "${TAGS[@]}"; do
    CKPT="results/checkpoints/ae_exp_masked_v2/${TAG}/ae_checkpoint_best.pt"
    echo "=== AE eval: ${TAG} ==="
    $PYTHON scripts/eval_ae_recon.py \
        --checkpoint "${CKPT}" \
        --mode atrophy \
        --figures-dir "ae_exp_masked_v2/${TAG}"
    echo "${TAG} done at: $(date)"
    echo ""
done

echo "All AE evals done. End: $(date)"
