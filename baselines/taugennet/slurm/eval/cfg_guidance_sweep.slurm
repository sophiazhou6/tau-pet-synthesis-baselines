#!/bin/bash
#SBATCH --job-name=cfg_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=results/logs/cfg_sweep_%j.out
#SBATCH --error=results/logs/cfg_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# CFG guidance-scale sweep on the CFG-trained combined model (legacy split).
#   w=0 -> unconditional (conditioning-sensitivity diagnostic)
#   w=1 -> standard conditional
#   w=3,5 -> amplified conditioning (does structure return?)
# Each w saves generations to an isolated dir; evaluated afterward on DK86 (correct SUVR).
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
mkdir -p results/logs
echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

for W in 0 1 3 5; do
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
