#!/bin/bash
# Stage 1: 5 AEs (latent_ch 2..6) each feeding 2 diffusion arms (small/large UNet).
# The diffusion arms wait on their own AE via afterok, so everything else runs in parallel.
cd "$(dirname "$0")/../.." || exit 1
for LCH in 2 3 4 5 6; do
  AEJ=$(sbatch --parsable slurm/sweep/s1_ae_lch${LCH}.slurm)
  echo "lch=${LCH}  AE=${AEJ}"
  for ARM in small large; do
    D=$(sbatch --parsable --dependency=afterok:${AEJ} slurm/sweep/s1_diff_lch${LCH}_${ARM}.slurm)
    echo "         diff ${ARM}=${D}"
  done
done
