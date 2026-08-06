#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=results/logs/cond_%x_%j.out
#SBATCH --error=results/logs/cond_%x_%j.err

# args: $1=SPATIAL_MAP  $2=cond_mode  $3=tag  $4=fold  $5=extra
MAP=$1; CM=$2; TAG=$3; FOLD=$4; EXTRA=${5:-}

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
export TAUGENNET_SPATIAL_MAP=$MAP
export TAUGENNET_REQUIRE_PTAU=1

echo "=== arm=$TAG fold=$FOLD map=$MAP cond=$CM extra='$EXTRA'  $(date) ==="
/usr/bin/python3.9 scripts/train_spatial_atrophy.py \
    --use-controls-322 --cond-mode $CM \
    --fold $FOLD --n-folds 5 \
    --skip-ae --ae-checkpoint results/checkpoints/sweep/ae_lch2/fold_0/taugennet_checkpoint_best.pt \
    --latent-ch 2 --ae-res-blocks 2 \
    --unet-channels 128,256,512 --n-transformer 1 \
    --checkpoint-dir results/checkpoints/controls_322/$TAG/fold_$FOLD \
    --diff-epochs 2000 --patience 500 --use-mask $EXTRA
echo "exit=$?  done: $(date)"
