#!/bin/bash
#SBATCH --job-name=relu_ptau217
#SBATCH --output=results/logs/relu_ptau217_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00

cd /scratch/network/sz3962/taugennet
source /home/sz3962/.conda/envs/taugennet/bin/activate

mkdir -p results/checkpoints/relu/ptau217 results/figures/monitor/relu/ptau217 results/logs

python -u scripts/train_relu.py \
  --mode ptau217 \
  --skip-ae \
  --ae-checkpoint results/checkpoints/taugennet_checkpoint_relu_v160.pt \
  --use-mask \
  --checkpoint-dir results/checkpoints/relu/ptau217 \
  --monitor-dir results/figures/monitor/relu/ptau217 \
  --diff-epochs 2000 \
  --patience 50 \
  --val-every 1
