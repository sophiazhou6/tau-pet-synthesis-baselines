#!/bin/bash
#SBATCH --job-name=ff_ae_kl1e-6
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G --time=06:00:00
#SBATCH --output=results/logs/ff_ae_kl1e-6_%j.out
#SBATCH --error=results/logs/ff_ae_kl1e-6_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet   # EDIT
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
OUT=results/checkpoints/sweep/ae_lch3_kl1e-6/fold_0; mkdir -p $OUT results/logs; set -e
$PY scripts/train.py --mode ptau217 --fold 0 --n-folds 5 --use-mentor-split --use-mask \
    --ae-epochs 500 --diff-epochs 0 --batch-size 4 \
    --latent-ch 3 --latent-scale 8 --ae-res-blocks 2 --kl-weight 1e-6 \
    --patience 50 --ae-val-every 5 --ae-checkpoint $OUT/ae.pt --checkpoint-dir $OUT
rm -f $OUT/ae.pt

bash scripts/publish_to_fac.sh
