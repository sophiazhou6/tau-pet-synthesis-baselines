#!/bin/bash
#SBATCH --job-name=relu_atrophy
#SBATCH --output=results/logs/relu_atrophy_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

cd /scratch/network/sz3962/taugennet
source /home/sz3962/.conda/envs/taugennet/bin/activate

mkdir -p results/checkpoints/relu/atrophy results/figures/monitor/relu/atrophy results/logs

python -u scripts/train_relu.py \
  --mode atrophy \
  --ae-epochs 300 \
  --ae-checkpoint results/checkpoints/taugennet_checkpoint_relu_v160.pt \
  --use-mask \
  --checkpoint-dir results/checkpoints/relu/atrophy \
  --monitor-dir results/figures/monitor/relu/atrophy \
  --diff-epochs 2000 \
  --patience 50 \
  --val-every 1
