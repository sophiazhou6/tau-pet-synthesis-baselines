#!/bin/bash
#SBATCH --job-name=canary_cfgm10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:15:00
#SBATCH --output=results/logs/canary_cfgm10_%j.out
#SBATCH --error=results/logs/canary_cfgm10_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
rm -rf results/_canary/cfgm10
echo "[canary] atrophy-spatial + ptau217_mlp + tissue — 2-epoch smoke test"
$PY scripts/train_spatial_atrophy.py --cond-mode ptau217_mlp --use-tissue --cfg-prob 0.15 \
    --fold 0 --n-folds 5 --use-mentor-split --use-mask --latent-ch 3 \
    --unet-channels 128,256,512 --n-transformer 1 \
    --skip-ae --ae-checkpoint results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt \
    --diff-epochs 2 --patience 1 --val-every 1 --checkpoint-dir results/_canary/cfgm10 \
  && echo "[canary] PASS -- combo trains cleanly" \
  || echo "[canary] FAIL -- combo has a bug (see traceback above)"
