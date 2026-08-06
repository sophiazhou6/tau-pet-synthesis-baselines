#!/bin/bash
#SBATCH --job-name=copy_imaging_scratch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/copy_imaging_%j.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/copy_imaging_%j.err
cd ~/taugennet
for d in 1mm_parcellated_CN_subj 1mm_parcellated_AD_subj 1mm_parcellated_MCI_subj cerebellumNormalized_AD_MCI; do
  if [ -L "data/raw/$d" ]; then
    echo "=== copying $d $(date) ==="
    rsync -a --stats "data/raw/$d/" "data/raw/_local_copies/$d/"
    SRC_COUNT=$(find "data/raw/$d" -type f | wc -l)
    DST_COUNT=$(find "data/raw/_local_copies/$d" -type f | wc -l)
    echo "$d: source $SRC_COUNT files, copied $DST_COUNT files"
    if [ "$SRC_COUNT" != "$DST_COUNT" ]; then
      echo "MISMATCH on $d -- do not repoint this one until investigated"
    fi
  fi
done
echo "=== all copies done $(date) ==="
