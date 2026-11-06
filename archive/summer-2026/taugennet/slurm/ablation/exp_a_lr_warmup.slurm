#!/bin/bash
#SBATCH --job-name=exp_a_warmup
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=7:00:00
#SBATCH --output=results/logs/exp_a_warmup_%A_%a.out
#SBATCH --error=results/logs/exp_a_warmup_%A_%a.err

# Experiment A: LR warmup only (100 epochs). Baseline = cond_b (spatial, cond-none, no EMA/min-SNR).

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
FOLD=$SLURM_ARRAY_TASK_ID
AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/lr_ablation/exp_a/fold_${FOLD}"
FIGDIR="results/figures/lr_ablation/exp_a/fold_${FOLD}"
mkdir -p "$OUTDIR" "$FIGDIR" results/logs

echo "Job $SLURM_JOB_ID  fold ${FOLD}/5  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON scripts/train_spatial_atrophy.py \
    --cond-mode none \
    --skip-ae --ae-checkpoint "${AE_CKPT}" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --fold $FOLD --n-folds 5 \
    --diff-epochs 2000 --patience 250 --val-every 5 \
    --lr-warmup-epochs 100 \
    --checkpoint-dir "${OUTDIR}" --figures-dir "${FIGDIR}" \
    --use-mask

rm -f "${OUTDIR}/diff_spatial_atrophy.pt"
echo "Done: $(date)"
