#!/bin/bash
#SBATCH --job-name=bl_b4
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G --time=08:00:00
#SBATCH --output=results/logs/bl_b4_%j.out
#SBATCH --error=results/logs/bl_b4_%j.err
# CLIP p-tau217 (cross-attention) + atrophy SPATIAL MAP (channel concat). Spatial family;
# reuses the SAME shared AE (models_spatial.Autoencoder3D is structurally identical).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet                   # EDIT
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
AE=results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt   # reuse completed pa_ae (lch3, scale8, rb2)
OUT=results/checkpoints/baselines/b4_clip_atrophy_spatialmap/fold_0
GEN=results/generated/baselines/b4_clip_atrophy_spatialmap/fold_0
REC=results/records/baselines/b4_clip_atrophy_spatialmap/fold_0
mkdir -p $OUT $GEN $REC results/logs
set -e
$PY scripts/train_spatial_atrophy.py --cond-mode ptau217 \
    --fold 0 --n-folds 5 --use-mentor-split --use-mask \
    --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 \
    --skip-ae --ae-checkpoint $AE \
    --diff-epochs 2000 --patience 250 --val-every 5 --checkpoint-dir $OUT
CKPT=$(ls -t $OUT/*best*.pt | head -1)
$PY scripts/generate_spatial.py --checkpoint "$CKPT" --generated-dir $GEN \
    --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --k 16 --fold 0 --n-folds 5 --use-mask --use-mentor-split
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 \
    --fold 0 --n-folds 5 --use-cached --generated-dir $GEN \
    --use-mask --use-mentor-split --skip-ablation \
    --figures-dir baselines/b4_clip_atrophy_spatialmap --records-dir $REC --metrics-out $REC/metrics.json
$PY scripts/localization_metrics.py $REC --sort-by peak
