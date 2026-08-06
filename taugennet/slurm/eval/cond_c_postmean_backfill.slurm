#!/bin/bash
#SBATCH --job-name=cc_pm_bf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/cc_pm_bf_%j.out
#SBATCH --error=results/logs/cc_pm_bf_%j.err

# SHORT backfill: cond_c posterior mean, K=8 over first 8 subjects. ~15 min so it slots
# into GPU gaps ahead of big jobs -> quick directional read on whether the posterior mean
# captures tau burden in the glass brain.
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
CKPT="results/checkpoints/cond_compare_maskfix/spatialmap_ptau/fold_0/diff_spatial_atrophy_best.pt"
GEN="results/generated/cond_c_postmean_k8_bf/fold_0"
REC="results/records/cond_c_postmean_k8_bf"
mkdir -p "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== generate posterior mean K=8, first 8 subjects ==="
$PY scripts/generate_spatial.py \
    --checkpoint "$CKPT" --generated-dir "$GEN" \
    --latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --k 8 --max-subjects 8 \
    --fold 0 --n-folds 5 --use-mask --no-use-mentor-split \
    || { echo "GEN FAILED"; exit 1; }

echo "=== evaluate (8 subjects) + glass brain (pearson-ranked) ==="
$PY scripts/evaluate_final.py \
    --mode atrophy --dataset spatial --cond-mode ptau217 \
    --fold 0 --n-folds 5 \
    --use-cached --generated-dir "$GEN" \
    --use-mask --no-use-mentor-split --skip-ablation --glass-brain \
    --figures-dir cond_c_postmean_k8_bf \
    --records-dir "$REC" --metrics-out "$REC/metrics.json"
echo "Done: $(date)"
