#!/bin/bash
#SBATCH --job-name=ae_kl
#SBATCH --partition=gpu --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --time=04:00:00
#SBATCH --output=results/logs/ae_kl_%j.out --error=results/logs/ae_kl_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
KL=$1
OUT=results/checkpoints/sweep/ae_lch3_kl${KL}/fold_0; mkdir -p $OUT
$PY scripts/train_spatial_sweep.py --latent-ch 3 --kl-weight ${KL} \
    --cond-mode atrophy --fold 0 --n-folds 5 --use-mentor-split --use-mask \
    --batch-size 4 --ae-epochs 100 --diff-epochs 0 --ae-checkpoint $OUT/taugennet_checkpoint.pt
