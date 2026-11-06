#!/bin/bash
#SBATCH --job-name=regen_cond_a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=results/logs/regen_cond_a_%j.out
#SBATCH --error=results/logs/regen_cond_a_%j.err

# Regenerate cond_a (mlp_atrophy) gens + DK86-masked glass brain, LEGACY split
# (evaluation uses legacy, per user). Isolated --generated-dir so no path collision.
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
GEN="results/generated/dk86_regen/cond_a_mlp/fold_0"
REC="results/records/dk86_glass/cond_a_mlp"
mkdir -p "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY scripts/evaluate_final.py \
    --mode atrophy --dataset final --fold 0 --n-folds 5 \
    --checkpoint-dir results/checkpoints/cond_compare/mlp_atrophy/fold_0/atrophy --use-best \
    --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --use-mask --no-use-mentor-split --skip-ablation \
    --glass-brain \
    --generated-dir "$GEN" \
    --figures-dir dk86_glass/cond_a_mlp \
    --records-dir "$REC" --metrics-out "$REC/metrics.json"
echo "Done: $(date)"
