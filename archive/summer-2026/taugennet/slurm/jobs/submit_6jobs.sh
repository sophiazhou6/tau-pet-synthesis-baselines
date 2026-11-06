#!/bin/bash
set -e
cd /scratch/network/sz3962/taugennet

# Both atrophy jobs start immediately — each trains its own masked AE then diffusion
JOB_SILU_ATR=$(sbatch job_silu_atrophy.slurm | awk '{print $4}')
JOB_RELU_ATR=$(sbatch job_relu_atrophy.slurm | awk '{print $4}')
echo "Atrophy jobs submitted: silu=${JOB_SILU_ATR}  relu=${JOB_RELU_ATR}"

# SiLU ptau217 and combined wait for SiLU AE (written by silu_atrophy)
JOB_SILU_PT=$(sbatch --dependency=afterok:${JOB_SILU_ATR} job_silu_ptau217.slurm  | awk '{print $4}')
JOB_SILU_COM=$(sbatch --dependency=afterok:${JOB_SILU_ATR} job_silu_combined.slurm | awk '{print $4}')
echo "SiLU follow-on jobs:   ptau217=${JOB_SILU_PT}  combined=${JOB_SILU_COM}"

# ReLU ptau217 and combined wait for ReLU AE (written by relu_atrophy)
JOB_RELU_PT=$(sbatch --dependency=afterok:${JOB_RELU_ATR} job_relu_ptau217.slurm  | awk '{print $4}')
JOB_RELU_COM=$(sbatch --dependency=afterok:${JOB_RELU_ATR} job_relu_combined.slurm | awk '{print $4}')
echo "ReLU follow-on jobs:   ptau217=${JOB_RELU_PT}  combined=${JOB_RELU_COM}"

echo ""
echo "Monitor:"
echo "  squeue -u sz3962"
echo "  tail -f results/logs/silu_atrophy_${JOB_SILU_ATR}.out"
echo "  tail -f results/logs/relu_atrophy_${JOB_RELU_ATR}.out"
