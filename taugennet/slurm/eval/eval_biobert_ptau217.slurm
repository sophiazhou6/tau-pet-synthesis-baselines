#!/bin/bash
#SBATCH --job-name=eval_biobert_ptau217
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/eval_biobert_ptau217_%j.out
#SBATCH --error=results/logs/eval_biobert_ptau217_%j.err

# Canonical eval (fresh inference) for the BiomedBERT p-tau217 run.
# Reuses evaluate_final.py; isolated generated/ namespace per the npy-isolation rule.

PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/figures/biobert/ptau217/eval_final \
         results/records/biobert \
         results/generated/biobert/ptau217 \
         results/logs

echo "=== eval_final: ptau217_biobert — $(date) ==="
$PYTHON scripts/evaluate_final.py \
    --mode ptau217_biobert \
    --checkpoint-dir results/checkpoints/biobert_ptau217 \
    --figures-dir biobert/ptau217/eval_final \
    --records-dir results/records/biobert \
    --arch silu \
    --use-mask \
    --n-steps 500 \
    --glass-brain \
    --save-generated \
    --generated-dir /home/remote/adebnath1/taugennet/results/generated/biobert/ptau217
echo "ptau217_biobert done at: $(date)"

echo "Evaluation complete. End time: $(date)"
