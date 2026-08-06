#!/bin/bash
#SBATCH --job-name=regen_small
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=00:35:00
#SBATCH --output=results/logs/regen_small_%j.out
#SBATCH --error=results/logs/regen_small_%j.err

# Regenerate small_arch combined gens + DK86-masked glass brain, LEGACY split.
# Isolated --generated-dir so no path collision with silu.
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
GEN="results/generated/dk86_regen/small_combined"
REC="results/records/dk86_glass/small_combined"
mkdir -p "$GEN" "$REC" results/logs
echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY scripts/evaluate_final.py \
    --mode combined --dataset combined \
    --checkpoint-dir results/checkpoints/small_arch --use-best \
    --unet-channels 128,256,512 --n-transformer 1 \
    --n-steps 500 --use-mask --no-use-mentor-split --skip-ablation \
    --glass-brain \
    --generated-dir "$GEN" \
    --figures-dir dk86_glass/small_combined \
    --records-dir "$REC" --metrics-out "$REC/metrics.json"
echo "Done: $(date)"
