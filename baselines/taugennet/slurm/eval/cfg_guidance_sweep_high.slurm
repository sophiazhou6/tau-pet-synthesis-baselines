#!/bin/bash
#SBATCH --job-name=cfg_hi
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/cfg_hi_%j.out
#SBATCH --error=results/logs/cfg_hi_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Extend the CFG guidance sweep: w=0/1/3/5 climbed monotonically (0.374->0.426, still rising).
# This finds where per-subject voxel R saturates (or degrades from over-guidance).
# Same CFG-trained combined model + legacy split; each w -> isolated gen dir, DK86 SUVR eval.
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
mkdir -p results/logs
echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

for W in 7 10 15; do
  echo "===== guidance_scale = $W ====="
  GEN="results/generated/cfg_sweep/combined_w${W}"
  $PY scripts/evaluate_cfg.py \
      --mode combined \
      --checkpoint-dir results/checkpoints/cfg/combined --use-best \
      --unet-channels 128,256,512 --n-transformer 1 \
      --n-steps 500 --use-mask \
      --guidance-scale ${W} \
      --generated-dir "$GEN" \
      --figures-dir cfg_sweep/combined_w${W} \
      --records-dir results/records/cfg_sweep/combined_w${W} \
      || echo "w=${W} FAILED"
done
echo "Done: $(date)"
