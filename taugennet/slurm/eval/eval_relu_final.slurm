#!/bin/bash
#SBATCH --job-name=eval_relu_final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/eval_relu_final_%j.out
#SBATCH --error=results/logs/eval_relu_final_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

mkdir -p results/figures/relu/atrophy/eval_final \
         results/figures/relu/ptau217/eval_final \
         results/figures/relu/combined/eval_final \
         results/records/relu_final \
         results/logs

for MODE in atrophy ptau217 combined; do
    echo "=== eval_final: $MODE (relu) — $(date) ==="
    $PYTHON scripts/evaluate_final.py \
        --mode $MODE \
        --checkpoint-dir results/checkpoints/relu/$MODE \
        --figures-dir relu/$MODE/eval_final \
        --records-dir results/records/relu_final \
        --arch relu \
        --use-mask \
        --n-steps 500 \
        --save-generated
    echo "$MODE done at: $(date)"
done

echo "All relu final evals complete. End: $(date)"
