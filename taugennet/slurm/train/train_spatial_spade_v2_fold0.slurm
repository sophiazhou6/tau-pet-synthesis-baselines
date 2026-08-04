#!/bin/bash
#SBATCH --job-name=spatial_spade_v2_fold0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/spatial_spade_v2_fold0_%j.out
#SBATCH --error=results/logs/spatial_spade_v2_fold0_%j.err

# CORRECTED full fold-0 run for spatial-injection Design A (SPADE), fixing all issues
# found in the v1 run (jobs 885659/885074):
#   - paper AE (latent_ch=3, n_res_blocks=2) instead of the wrong ae_lch8_dk86 (latent_ch=8)
#   - --cond-mode ptau217 to match cfgm_5 exactly (spatial map + ptau217 cross-attention)
#   - larger UNet (256,512,768 / n_transformer=3) matching c04, the sweep's current leader
# Baseline to beat (CV-mean, NOT the inflated single-fold 0.510): cfgm_5 SET3=0.389/SET1=0.336;
# best-arch c04 SET3=0.466/SET1=0.336. This is fold-0 only — 5-fold it if it looks promising.

set -e

PYTHON=/usr/bin/python3.9
cd $HOME/taugennet
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:$PYTHONPATH
export TAUGENNET_ROOT=$HOME/taugennet

AE_CKPT="results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt"
OUTDIR="results/checkpoints/spatial_spade_v2/fold_0"
FIGDIR="results/figures/spatial_spade_v2/fold_0"
GENDIR="results/generated/spatial_spade_v2/ptau217/fold_0"
RECDIR="results/records/spatial_spade_v2/ptau217/fold_0"
mkdir -p "$OUTDIR" "$FIGDIR" "$GENDIR" "$RECDIR" results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/3] Training ==="
$PYTHON scripts/train_spatial_atrophy.py \
    --skip-ae \
    --ae-checkpoint "${AE_CKPT}" \
    --latent-ch 3 \
    --ae-res-blocks 2 \
    --unet-channels 256,512,768 \
    --n-transformer 3 \
    --diff-epochs 2000 \
    --patience 250 \
    --val-every 5 \
    --fold 0 \
    --n-folds 5 \
    --use-mentor-split \
    --use-mask \
    --cond-mode ptau217 \
    --use-spade \
    --spade-hidden 64 \
    --checkpoint-dir "${OUTDIR}" \
    --figures-dir "${FIGDIR}"

echo "=== [2/3] Generating (DDPM, 500 steps, fold-0 held-out test set) ==="
$PYTHON scripts/generate_spatial.py \
    --checkpoint "${OUTDIR}/diff_spatial_atrophy_best.pt" \
    --generated-dir "${GENDIR}" \
    --latent-ch 3 \
    --unet-channels 256,512,768 \
    --n-transformer 3 \
    --n-steps 500 \
    --fold 0 \
    --n-folds 5 \
    --use-mentor-split \
    --use-mask

echo "=== [3/3] Evaluating (SET1/2/3 Pearson + glass brain, SUVR, DK86-masked) ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy \
    --dataset spatial \
    --cond-mode ptau217 \
    --use-cached \
    --generated-dir "${GENDIR}" \
    --use-mentor-split \
    --glass-brain \
    --figures-dir spatial_spade_v2/ptau217/fold_0 \
    --records-dir "${RECDIR}"

echo "Publishing to FAC…"
bash scripts/publish_to_fac.sh

echo "Done: $(date)"
