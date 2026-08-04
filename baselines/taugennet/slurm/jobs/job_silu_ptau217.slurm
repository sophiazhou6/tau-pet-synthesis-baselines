#!/bin/bash
#SBATCH --job-name=silu_ptau217
#SBATCH --output=results/logs/silu_ptau217_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

cd /scratch/network/sz3962/taugennet
source /home/sz3962/.conda/envs/taugennet/bin/activate

mkdir -p results/checkpoints/silu/ptau217 results/figures/monitor/silu/ptau217 results/logs

python -u scripts/train.py \
  --mode ptau217 \
  --batch-size 4 \
  --skip-ae \
  --ae-checkpoint results/checkpoints/taugennet_checkpoint_silu_masked.pt \
  --use-mask \
  --checkpoint-dir results/checkpoints/silu/ptau217 \
  --monitor-dir results/figures/monitor/silu/ptau217 \
  --diff-epochs 2000 \
  --patience 50 \
  --val-every 1
