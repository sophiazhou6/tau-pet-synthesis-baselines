#!/bin/bash
#SBATCH --job-name=gae_s8d5
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/gae_s8d5_%A_%a.out
#SBATCH --error=results/logs/gae_s8d5_%A_%a.err

# AE-structure grid (focused on DEPTH, the paper-validated lever).
# latent scale=8 (latent 12x14x12) x depth=5 resblocks/stage.
# Fixed: latent_ch=6, kl=1e-5, DK86 mask, MENTOR split. AE-ONLY (--diff-epochs 0).
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
FOLD=$SLURM_ARRAY_TASK_ID
OUT=results/checkpoints/grid_ae_struct/s8_d5/fold_${FOLD}
AE_CKPT=$OUT/taugennet_checkpoint.pt
AE_BEST=$OUT/taugennet_checkpoint_best.pt
METRICS=$OUT/recon_metrics.json
mkdir -p $OUT results/logs
echo "=== AE grid scale=8 depth=5 fold ${FOLD}/5 (latent 12x14x12) ==="
echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
set -e
$PY scripts/train.py --mode atrophy --fold $FOLD --n-folds 5 --use-mentor-split --use-mask \
    --ae-epochs 500 --diff-epochs 0 \
    --latent-ch 6 --kl-weight 1e-5 \
    --latent-scale 8 --ae-res-blocks 5 \
    --patience 50 --ae-val-every 5 \
    --ae-checkpoint $AE_CKPT --checkpoint-dir $OUT
$PY scripts/eval_ae_recon.py --checkpoint $AE_BEST --mode atrophy \
    --fold $FOLD --n-folds 5 --split val --use-mask --no-figures \
    --metrics-out $METRICS
rm -f $AE_CKPT
echo "Fold ${FOLD} done: $(date)"
