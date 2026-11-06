#!/bin/bash
#SBATCH --job-name=postmean_a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/postmean_a_%j.out
#SBATCH --error=results/logs/postmean_a_%j.err

# Diffusion posterior mean (K=16 samples/subject) for cond_a atrophy, then eval
# (voxel Pearson + glass brain). Tests if averaging recovers DenseUNet-like fidelity.
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
GEN="results/generated/posterior_mean/cond_a_mlp/fold_0"
REC="results/records/posterior_mean/cond_a_mlp"
mkdir -p "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== Step 1: generate K=16 posterior mean ==="
$PY scripts/generate_posterior_mean.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/cond_compare/mlp_atrophy/fold_0/atrophy --use-best \
    --unet-channels 128,256,512 --n-transformer 1 --arch silu \
    --fold 0 --n-folds 5 --k 16 --n-steps 500 \
    --out-dir "$GEN" || { echo "GEN FAILED"; exit 1; }

echo "=== Step 2: evaluate posterior mean (voxel Pearson + glass brain, correct SUVR) ==="
$PY scripts/evaluate_final.py \
    --mode atrophy --dataset final --fold 0 --n-folds 5 \
    --use-cached --generated-dir "$GEN" \
    --use-mask --no-use-mentor-split --skip-ablation --glass-brain \
    --figures-dir posterior_mean/cond_a_mlp \
    --records-dir "$REC" --metrics-out "$REC/metrics.json"
echo "Done: $(date)"
