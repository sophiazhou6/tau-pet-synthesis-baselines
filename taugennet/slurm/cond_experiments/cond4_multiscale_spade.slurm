#!/bin/bash
#SBATCH --job-name=cond_spade
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/cond_spade_%j.out
#SBATCH --error=results/logs/cond_spade_%j.err

# Experiment 4: MULTI-SCALE SPADE injection — atrophy(+tissue) modulates EVERY UNet stage,
# targeting WHERE tau burden goes (spatial localization). New arch -> trains from scratch (diffusion).
# dataset_spatial, cond-mode ptau217 + --use-tissue + --use-spade. Spatial-family AE (lch8).
PY=/usr/bin/python3.9
AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
OUT="results/checkpoints/cond4_spade/fold_0"
GEN="results/generated/cond4_spade/fold_0"; REC="results/records/cond4_spade/fold_0"
mkdir -p "$OUT" "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  Start: $(date)"
$PY scripts/train_spatial_atrophy.py --cond-mode ptau217 --use-tissue --use-spade \
    --fold 0 --n-folds 5 --use-mentor-split --use-mask \
    --skip-ae --ae-checkpoint "$AE_CKPT" \
    --unet-channels 256,512,768 --n-transformer 3 \
    --diff-epochs 2000 --patience 250 --val-every 5 --checkpoint-dir "$OUT" \
    || { echo TRAIN FAILED; exit 1; }
CKPT=$(ls -t "$OUT"/*best*.pt 2>/dev/null | head -1)
$PY scripts/generate_spatial.py --checkpoint "$CKPT" --generated-dir "$GEN" \
    --latent-ch 8 --unet-channels 256,512,768 --n-transformer 3 \
    --n-steps 500 --fold 0 --n-folds 5 --use-mask --use-mentor-split
# add --k 16 above for the posterior mean (best for the WHERE-does-tau-go read)
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 \
    --fold 0 --n-folds 5 --use-cached --generated-dir "$GEN" \
    --use-mask --use-mentor-split --skip-ablation --glass-brain \
    --figures-dir cond4_spade --records-dir "$REC" --metrics-out "$REC/metrics.json"
echo "Done: $(date)"
