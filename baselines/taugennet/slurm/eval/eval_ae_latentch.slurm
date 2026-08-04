#!/bin/bash
#SBATCH --job-name=eval_ae_lch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=results/logs/eval_ae_lch_%j.out
#SBATCH --error=results/logs/eval_ae_lch_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
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
