#!/bin/bash
#SBATCH --job-name=silu_atrophy
#SBATCH --output=results/logs/silu_atrophy_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
source /home/remote/adebnath1/.conda/envs/taugennet/bin/activate

mkdir -p results/checkpoints/silu/atrophy results/figures/monitor/silu/atrophy results/logs

python -u scripts/train.py \
  --mode atrophy \
  --batch-size 4 \
  --ae-epochs 300 \
  --ae-checkpoint results/checkpoints/taugennet_checkpoint_silu_masked.pt \
  --use-mask \
  --checkpoint-dir results/checkpoints/silu/atrophy \
  --monitor-dir results/figures/monitor/silu/atrophy \
  --diff-epochs 2000 \
  --patience 50 \
  --val-every 1
