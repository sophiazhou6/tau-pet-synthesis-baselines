#!/bin/bash
#SBATCH --job-name=bl_b3_combined_clip_atrophyMLP
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G --time=08:00:00
#SBATCH --output=results/logs/bl_b3_combined_clip_atrophyMLP_%j.out
#SBATCH --error=results/logs/bl_b3_combined_clip_atrophyMLP_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet                   # EDIT
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
AE=results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt   # reuse completed pa_ae (lch3, scale8, rb2)
OUT=results/checkpoints/baselines/b3_combined_clip_atrophyMLP/fold_0
GEN=results/generated/baselines/b3_combined_clip_atrophyMLP/fold_0
REC=results/records/baselines/b3_combined_clip_atrophyMLP/fold_0
mkdir -p $OUT $GEN $REC results/logs
set -e
$PY scripts/train.py --mode combined --fold 0 --n-folds 5 --use-mentor-split --use-mask \
    --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 \
    --skip-ae --ae-checkpoint $AE \
    --diff-epochs 2000 --patience 250 --val-every 5 --checkpoint-dir $OUT
$PY scripts/generate_posterior_mean.py --mode combined --checkpoint-dir $OUT --use-best \
    --k 16 --n-steps 500 --unet-channels 128,256,512 --n-transformer 1 --arch silu \
    --fold 0 --n-folds 5 --use-mentor-split --use-mask --out-dir $GEN
$PY scripts/evaluate_final.py --mode combined --dataset combined --fold 0 --n-folds 5 \
    --use-cached --generated-dir $GEN --use-mask --use-mentor-split --skip-ablation \
    --figures-dir baselines/b3_combined_clip_atrophyMLP --records-dir $REC --metrics-out $REC/metrics.json
$PY scripts/localization_metrics.py $REC --sort-by peak
