#!/bin/bash
#SBATCH --job-name=reg_t1
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=07:59:00
#SBATCH --output=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/regt1_%j.out
#SBATCH --error=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/regt1_%j.err

# All 597 subjects of pinned_scans_597.csv in ONE job (the short QOS submit cap is full):
# 32 subjects at a time, ~6 min each → ~2 h expected. FLIRT + FNIRT with the lab's settings →
# $TAUGENNET_ROOT/data/mni_t1/<PTID>/T1_to_MNI_nonlin.nii.gz. Resumable: done subjects skip in
# seconds, so resubmitting after any failure only redoes what is missing.
set -uo pipefail
export TAUGENNET_ROOT="${TAUGENNET_ROOT:-/scratch/network/sz3962/taugennet}"
USED=$( (checkquota 2>/dev/null || true) | awk '$0 ~ "/scratch/network" && $0 ~ "GiB" {v=$5; sub("GiB","",v); print v; exit}')
if [ -n "${USED:-}" ] && awk -v u="$USED" -v m="${QUOTA_MAX_GIB:-185}" 'BEGIN{exit !(u>m)}'; then
  echo "ABORT: /scratch/network at ${USED} GiB (> ${QUOTA_MAX_GIB:-185})"; exit 3
fi
seq 0 596 | xargs -P 32 -I{} sh -c \
  '/scratch/network/sz3962/.conda/envs/fslreg/bin/python3 /scratch/network/sz3962/tau-denseunet-baseline/scripts/register_pinned_t1.py --index {} || echo "FAILED_INDEX {}"'
N=$(find "$TAUGENNET_ROOT/data/mni_t1" -name T1_to_MNI_nonlin.nii.gz | wc -l)
echo "registered T1s on disk: $N / 597"
[ "$N" -ge 597 ]
