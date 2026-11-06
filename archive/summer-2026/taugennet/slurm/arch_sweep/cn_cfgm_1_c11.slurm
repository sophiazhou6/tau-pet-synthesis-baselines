#!/bin/bash
#SBATCH --job-name=cn_cfgm_1_c11
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --mem=64G --time=00:20:00
#SBATCH --output=results/logs/cn_cfgm_1_c11_%A_%a.out
#SBATCH --error=results/logs/cn_cfgm_1_c11_%A_%a.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
set -e
BASE=arch_sweep_canary/cfgm_1/c11
OUT=results/checkpoints/$BASE; rm -rf $OUT; mkdir -p $OUT results/logs
$PY scripts/train_spatial_sweep.py --cond-mode none --cfg-prob 0.15 --fold 0 --n-folds 5 \
    --use-mentor-split --use-mask --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 \
    --noise-schedule linear --skip-ae --ae-checkpoint results/checkpoints/sweep/ae_lch3_kl1e-3/fold_0/taugennet_checkpoint_best.pt \
    --diff-epochs 2 --patience 1 --val-every 1 --monitor-every 999999 --checkpoint-dir $OUT
CKPT=$(ls -t $OUT/*best*.pt | head -1)
G=results/generated/$BASE/w1; mkdir -p $G
$PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G --latent-ch 3 \
    --unet-channels 128,256,512 --n-transformer 1 --n-steps 10 --guidance-scale 1 --k 1 --fold 0 --n-folds 5 --use-mask --use-mentor-split
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode atrophy --fold 0 --n-folds 5 \
    --use-cached --generated-dir $G --use-mask --use-mentor-split --skip-ablation --no-glass-brain \
    --records-dir results/records/$BASE/w1 && echo "[canary] PASS cfgm_1/c11" || echo "[canary] FAIL cfgm_1/c11"
