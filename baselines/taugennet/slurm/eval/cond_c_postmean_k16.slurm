#!/bin/bash
#SBATCH --job-name=cc_postmean
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/cc_postmean_%j.out
#SBATCH --error=results/logs/cc_postmean_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# cond_c POSTERIOR MEAN: average K=16 DDPM samples/subject (existing checkpoint, no retrain).
# Tests whether the conditional mean captures spatial tau burden (vs the misaligned single sample).
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
CKPT="results/checkpoints/cond_compare_maskfix/spatialmap_ptau/fold_0/diff_spatial_atrophy_best.pt"
GEN="results/generated/cond_c_postmean_k16/fold_0"
REC="results/records/cond_c_postmean_k16"
mkdir -p "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== Step 1: generate posterior mean (K=16, legacy fold-0) ==="
$PY scripts/generate_spatial.py \
    --checkpoint "$CKPT" --generated-dir "$GEN" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --k 16 --fold 0 --n-folds 5 --use-mask --no-use-mentor-split \
    || { echo "GEN FAILED"; exit 1; }

echo "=== Step 2: evaluate posterior mean (DK86, correct SUVR) + glass brain (pearson-ranked) ==="
$PY scripts/evaluate_final.py \
    --mode atrophy --dataset spatial --cond-mode ptau217 \
    --fold 0 --n-folds 5 \
    --use-cached --generated-dir "$GEN" \
    --use-mask --no-use-mentor-split --skip-ablation --glass-brain \
    --figures-dir cond_c_postmean_k16 \
    --records-dir "$REC" --metrics-out "$REC/metrics.json"
echo "Done: $(date)"
