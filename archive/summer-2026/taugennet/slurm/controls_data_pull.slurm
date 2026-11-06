#!/bin/bash
#SBATCH --job-name=controls_data_pull
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/controls_pull_%j.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/controls_pull_%j.err
set -uo pipefail
cd ~/taugennet

rm -rf ~/taugennet/data/raw/_fac_pull_test   # canary artifact, superseded by this real pull

DEST=~/taugennet/data/raw/controls_322
mkdir -p "$DEST"

echo "=== pulling CN MRI dir (~4.2G) ==="
timeout 1800 rsync -avh --stats /mnt/fac/CX500024_DS2/ADNI_Data_INR/data/1mm_parcellated_CN_subj "$DEST/"

echo "=== pulling combined PET dir (~15G) ==="
timeout 1800 rsync -avh --stats /mnt/fac/CX500024_DS2/ADNI_Data_INR/data/cerebellumNormalized_AD_MCI_CN "$DEST/"

echo "=== pulling split CSVs + SUVR CSV ==="
rsync -avh /mnt/fac/CX500024_DS2/ADNI_Data_INR/train_val_split.csv \
    /mnt/fac/CX500024_DS2/ADNI_Data_INR/heldout_test_split.csv \
    /mnt/fac/CX500024_DS2/ADNI_Data_INR/regional_SUVR_cerebellumNormalizedGTScan1.csv \
    "$DEST/"

echo "=== verify ==="
du -sh "$DEST"/1mm_parcellated_CN_subj "$DEST"/cerebellumNormalized_AD_MCI_CN
find "$DEST"/1mm_parcellated_CN_subj -mindepth 1 -maxdepth 1 -type d | wc -l
ls -la "$DEST"/*.csv
