#!/bin/bash
# FIRE-AND-FORGET grid search — queues EVERYTHING in one command. No check-ins needed.
# Uses a FIXED conditioning baseline (COND_MODE below) so it does NOT wait on the conditioning
# matrix. All CFG + mentor split + paper-spec AE (rb=2). Rank later with rank_sweep.sh.
#
# BEFORE RUNNING: edit PY= and cd-path in every slurm/sweep/*.slurm (there's a sed one-liner in
# FIRE_AND_FORGET.md), and set COND_MODE here to your chosen baseline.
set -e
cd "$(dirname "$0")/../.." || exit 1
COND_MODE=combined_mlp          # fixed baseline (atrophy MLP + CLIP p-tau). combined_mlp also valid.

echo "=== Stage 1: latent_ch {2,3,4,5,6} x UNet {small,large}  (5 AE + 10 CFG diffusion) ==="
declare -A AEJ
for LCH in 2 3 4 5 6; do
  AEJ[$LCH]=$(sbatch --parsable slurm/sweep/s1_ae_lch${LCH}.slurm)
  echo "  AE lch=$LCH = ${AEJ[$LCH]}"
  for ARM in small large; do
    D=$(sbatch --parsable --dependency=afterok:${AEJ[$LCH]} slurm/sweep/s1_diff_lch${LCH}_${ARM}.slurm)
    echo "     diff ${ARM} = $D"
  done
done

echo "=== kl {1e-3,1e-4,1e-6} at lch3-small  (3 AE + 3 diffusion; 1e-5 = the stage-1 lch3 cell) ==="
for KL in 1e-3 1e-4 1e-6; do
  J=$(sbatch --parsable slurm/sweep/ff_ae_lch3_kl${KL}.slurm)
  D=$(sbatch --parsable --dependency=afterok:$J slurm/sweep/ff_diff_lch3_kl${KL}.slurm)
  echo "  kl=$KL  AE=$J  diff=$D"
done

echo "=== cosine noise schedule at lch3-small (reuses the stage-1 lch3 AE) ==="
sbatch --dependency=afterok:${AEJ[3]} slurm/sweep/ff_diff_lch3_cosine.slurm

echo
echo "DONE — ~22 jobs queued, all dependency-chained. Walk away."
echo "When you're back:  bash slurm/sweep/rank_sweep.sh ${COND_MODE}"
echo "  -> SET1/SET2/SET3/WITHIN/PEAK across every cell, from cached generations."
