#!/bin/bash
#SBATCH --job-name=condb_regen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --output=results/logs/condb_regen_%j.out
#SBATCH --error=results/logs/condb_regen_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Re-generate cond_b on its OWN legacy held-out test set (the split it was TRAINED on -> no
# leakage), SEEDED for reproducibility, then evaluate + render glass brains from the SAME
# volumes so the Regional R and the glass brains are guaranteed self-consistent.
#   Pass 1: single sample (k=1)  -> reproducible analog of the 0.586 (single-draw) methodology
#   Pass 2: posterior mean (k=10) -> clean maps, no salt-and-pepper

PY=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
CKPT="results/checkpoints/cond_compare_maskfix/spatialmap_none/fold_0/diff_spatial_atrophy_best.pt"
ARGS="--latent-ch 8 --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 --fold 0 --n-folds 5 --no-use-mentor-split --seed 0"
echo "Job $SLURM_JOB_ID  Node $SLURM_NODELIST  Start $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== Pass 1: single-sample (k=1), seeded ==="
GEN1="results/generated/cond_b_regen/single/fold_0"
$PY scripts/generate_spatial.py --checkpoint "$CKPT" --generated-dir "$GEN1" $ARGS --k 1 \
    && $PY scripts/regen_condb_check.py --generated-dir "$GEN1" \
           --out-dir results/figures/cond_b_regen/single --tag single

echo "=== Pass 2: posterior mean (k=10), seeded — clean glass brains ==="
GEN2="results/generated/cond_b_regen/pmean_k10/fold_0"
$PY scripts/generate_spatial.py --checkpoint "$CKPT" --generated-dir "$GEN2" $ARGS --k 10 \
    && $PY scripts/regen_condb_check.py --generated-dir "$GEN2" \
           --out-dir results/figures/cond_b_regen/pmean --tag pmean

echo "Done $(date)"
