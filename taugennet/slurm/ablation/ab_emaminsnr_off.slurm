#!/bin/bash
#SBATCH --job-name=ab_emasnr_off
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/ab_emasnr_off_%A_%a.out
#SBATCH --error=results/logs/ab_emasnr_off_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# A/B ablation — ARM A (without EMA / min-SNR). Baseline atrophy diffusion on the
# fixed ae_lch8_dk86 AE, 5-fold CV. Same as ARM B but with the upgrades OFF.

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
FOLD=$SLURM_ARRAY_TASK_ID
AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/ablation_emaminsnr/off/fold_${FOLD}/atrophy"
FIGDIR="results/figures/ablation_emaminsnr/off/fold_${FOLD}"
mkdir -p "$OUTDIR" "$FIGDIR" results/logs

echo "Job $SLURM_JOB_ID  fold ${FOLD}/5  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON scripts/train.py \
    --mode atrophy \
    --skip-ae --ae-checkpoint "${AE_CKPT}" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --fold $FOLD --n-folds 5 \
    --diff-epochs 2000 --patience 250 --val-every 5 \
    --checkpoint-dir "${OUTDIR}" --figures-dir "${FIGDIR}" \
    --use-mask

rm -f "${OUTDIR}/diff_atrophy.pt"
echo "Done: $(date)"
