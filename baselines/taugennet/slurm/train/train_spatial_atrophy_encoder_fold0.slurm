#!/bin/bash
#SBATCH --job-name=spatial_atrophy_encoder_fold0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/spatial_atrophy_encoder_fold0_%j.out
#SBATCH --error=results/logs/spatial_atrophy_encoder_fold0_%j.err

# Full fold-0 training + generation + evaluation for spatial-injection Design B
# (learned AtrophyEncoder3D). Canary (job 884919/885074-era) confirmed this code
# path works end-to-end; this is the real run. Baseline to beat: cfgm_5 SET3=0.510
# (per SPATIAL_INJECTION_CONTEXT.md).
#
# CoreHPC env confirmed 2026-07-14/15: python3.9 (no conda), gpu:nvidia_h100_nvl:1
# on partition=gpu, HOME=/home/remote/adebnath1. AE checkpoint is ae_lch8_dk86
# (latent_ch=8) — the paper_aligned/ae/fold_0 checkpoint does NOT match this
# repo's current Autoencoder3D architecture (confirmed via canary failure
# 2026-07-15), do not switch back to it without re-verifying compatibility first.

set -e   # fail loudly; also means generate/eval below are skipped if training fails

PYTHON=/usr/bin/python3.9
cd $HOME/taugennet
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:$PYTHONPATH
export TAUGENNET_ROOT=$HOME/taugennet

AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/spatial_atrophy_encoder/fold_0"
FIGDIR="results/figures/spatial_atrophy_encoder/fold_0"
GENDIR="results/generated/spatial_atrophy_encoder/atrophy/fold_0"
RECDIR="results/records/spatial_atrophy_encoder/atrophy/fold_0"
mkdir -p "$OUTDIR" "$FIGDIR" "$GENDIR" "$RECDIR" results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/3] Training ==="
$PYTHON scripts/train_spatial_atrophy.py \
    --skip-ae \
    --ae-checkpoint "${AE_CKPT}" \
    --latent-ch 8 \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --diff-epochs 2000 \
    --patience 250 \
    --fold 0 \
    --n-folds 5 \
    --use-mentor-split \
    --use-mask \
    --use-atrophy-encoder \
    --atrophy-encoder-width 32 \
    --checkpoint-dir "${OUTDIR}" \
    --figures-dir "${FIGDIR}"

echo "=== [2/3] Generating (DDPM, 500 steps, fold-0 held-out test set) ==="
$PYTHON scripts/generate_spatial.py \
    --checkpoint "${OUTDIR}/diff_spatial_atrophy_best.pt" \
    --generated-dir "${GENDIR}" \
    --latent-ch 8 \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --n-steps 500 \
    --fold 0 \
    --n-folds 5 \
    --use-mentor-split \
    --use-mask

echo "=== [3/3] Evaluating (SET1/2/3 Pearson + glass brain, SUVR, DK86-masked) ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy \
    --dataset spatial \
    --cond-mode atrophy \
    --use-cached \
    --generated-dir "${GENDIR}" \
    --use-mentor-split \
    --glass-brain \
    --figures-dir spatial_atrophy_encoder/atrophy/fold_0 \
    --records-dir "${RECDIR}"

echo "Publishing to FAC…"
bash scripts/publish_to_fac.sh

echo "Done: $(date)"
