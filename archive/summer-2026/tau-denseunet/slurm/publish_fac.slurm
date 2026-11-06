#!/bin/bash
#SBATCH --job-name=denseunet_publish
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/publish_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/publish_%j.err
set -euo pipefail
REPO=/home/remote/adebnath1/tau-denseunet-baseline
cd "$REPO"
source "$HOME/taugennet/scripts/wait_for_fac.sh"
bash "$HOME/taugennet/scripts/publish_to_fac.sh"
