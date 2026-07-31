#!/bin/bash
#SBATCH --job-name=cond_a_mlp
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/cond_a_mlp_%A_%a.out
#SBATCH --error=results/logs/cond_a_mlp_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Conditioning comparison — variant (a): atrophy as MLP cross-attention only
# (no spatial map). Non-spatial train.py. 5-fold CV on the fixed ae_lch8_dk86 AE.
# Same recipe as variants (b)/(c): EMA/min-SNR OFF, lch8, --use-mask.

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
FOLD=$SLURM_ARRAY_TASK_ID
AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/cond_compare/mlp_atrophy/fold_${FOLD}/atrophy"
FIGDIR="results/figures/cond_compare/mlp_atrophy/fold_${FOLD}"
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
