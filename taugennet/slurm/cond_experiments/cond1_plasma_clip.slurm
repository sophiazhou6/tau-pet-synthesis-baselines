#!/bin/bash
#SBATCH --job-name=cond_clip
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/cond_clip_%j.out
#SBATCH --error=results/logs/cond_clip_%j.err

# Experiments 1 (plasma CLIP) + 2 (plasma NO-MRI) in one job.
# dataset_final, mode ptau217 (scalar plasma via CLIP text; MRI structural via z_m).
# Dropping --skip-ablation makes evaluate_final produce BOTH "MRI+cond" and "cond-only" tables.
PY=/usr/bin/python3.9
AE_CKPT="results/checkpoints/grid_search/step0/lch6_kl1e-05/fold_0/taugennet_checkpoint_best.pt"  # lch6 (final-family AE); EDIT to grid winner
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
OUT="results/checkpoints/cond1_plasma_clip/fold_0"
GEN="results/generated/cond1_plasma_clip/fold_0"; REC="results/records/cond1_plasma_clip/fold_0"
mkdir -p "$OUT" "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  Start: $(date)"
$PY scripts/train.py --mode ptau217 --fold 0 --n-folds 5 --use-mentor-split --use-mask \
    --latent-ch 6 --unet-channels 256,512,768 --n-transformer 3 \
    --skip-ae --ae-checkpoint "$AE_CKPT" \
    --diff-epochs 2000 --patience 250 --val-every 5 --checkpoint-dir "$OUT" \
    || { echo TRAIN FAILED; exit 1; }
# NOTE: NO --skip-ablation -> runs the plasma-no-MRI (cond-only) comparison automatically
$PY scripts/evaluate_final.py --mode ptau217 --dataset final --cond-mode ptau217 \
    --checkpoint-dir "$OUT" --use-best --unet-channels 256,512,768 --n-transformer 3 --arch silu \
    --fold 0 --n-folds 5 --save-generated --generated-dir "$GEN" \
    --use-mask --use-mentor-split --glass-brain \
    --figures-dir cond1_plasma_clip --records-dir "$REC" --metrics-out "$REC/metrics.json"
echo "Done: $(date)"
