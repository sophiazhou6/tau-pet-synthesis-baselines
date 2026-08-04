#!/bin/bash
#SBATCH --job-name=relu_combined
#SBATCH --output=results/logs/relu_combined_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00

cd /scratch/network/sz3962/taugennet
source /home/sz3962/.conda/envs/taugennet/bin/activate

mkdir -p results/checkpoints/relu/combined results/figures/monitor/relu/combined results/logs

python -u scripts/train_relu.py \
  --mode combined \
  --skip-ae \
  --ae-checkpoint results/checkpoints/taugennet_checkpoint_relu_v160.pt \
  --use-mask \
  --checkpoint-dir results/checkpoints/relu/combined \
  --monitor-dir results/figures/monitor/relu/combined \
  --diff-epochs 2000 \
  --patience 50 \
  --val-every 1
