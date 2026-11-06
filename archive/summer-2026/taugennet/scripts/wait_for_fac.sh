#!/bin/bash
FAC_CHECK="${FAC_CHECK:-/mnt/fac/CX500024_DS2/ADNI_Data_INR/data/cerebellumNormalized_AD_MCI/AD}"
for i in $(seq 1 30); do ls "$FAC_CHECK" >/dev/null 2>&1 && break; echo "waiting for FAC mount ($i)..."; sleep 5; done
ls "$FAC_CHECK" >/dev/null 2>&1 || { echo "FAC not available on $(hostname); requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
echo "FAC mounted OK on $(hostname)"
