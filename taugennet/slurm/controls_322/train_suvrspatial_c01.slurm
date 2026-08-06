#!/bin/bash
#SBATCH --job-name=train_suvrspatial_c322
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/logs/train_suvrspatial_c322_%j.out
#SBATCH --error=results/logs/train_suvrspatial_c322_%j.err

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
export TAUGENNET_SPATIAL_MAP=suvr

echo "=== spatial SUVR-map + controls_322, small arch (matches suvrcond_c01) — $(date) ==="
/usr/bin/python3.9 scripts/train_spatial_atrophy.py \
    --use-controls-322 \
    --cond-mode none \
    --skip-ae --ae-checkpoint results/checkpoints/sweep/ae_lch2/fold_0/taugennet_checkpoint_best.pt \
    --latent-ch 2 --ae-res-blocks 2 \
    --unet-channels 128,256,512 --n-transformer 1 \
    --checkpoint-dir results/checkpoints/controls_322/suvrspatial_c01 \
    --diff-epochs 2000 --patience 500 \
    --use-mask
echo "exit=$?  done: $(date)"
