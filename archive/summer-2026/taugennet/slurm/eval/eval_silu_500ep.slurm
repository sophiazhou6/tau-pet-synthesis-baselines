#!/bin/bash
#SBATCH --job-name=eval_silu_500ep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/eval_silu_500ep_%j.out
#SBATCH --error=results/logs/eval_silu_500ep_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/figures/silu_500ep/atrophy/eval_final_v3 \
         results/figures/silu_500ep/ptau217/eval_final_v3 \
         results/figures/silu_500ep/combined/eval_final_v3 \
         results/records/silu_500ep \
         results/generated/silu_500ep/atrophy \
         results/generated/silu_500ep/ptau217 \
         results/generated/silu_500ep/combined \
         results/logs

for MODE in atrophy ptau217 combined; do
    echo "=== eval_500ep: $MODE (silu) — $(date) ==="
    $PYTHON scripts/evaluate_final.py \
        --mode $MODE \
        --checkpoint-dir results/checkpoints/silu/${MODE}_500ep \
        --figures-dir silu_500ep/$MODE/eval_final_v3 \
        --records-dir results/records/silu_500ep \
        --arch silu \
        --use-mask \
        --n-steps 500 \
        --save-generated \
        --generated-dir /home/remote/adebnath1/taugennet/results/generated/silu_500ep/$MODE
    echo "$MODE done at: $(date)"
done

echo "All silu 500ep evals complete. End: $(date)"
