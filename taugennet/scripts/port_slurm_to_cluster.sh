#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; SLURM_DIR="$ROOT_DIR/slurm"
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
: "${TAUGENNET_ROOT:?source cluster.env first}"; : "${TAUGENNET_CONDA_ENV:?}"; : "${TAUGENNET_PARTITION:?}"; : "${TAUGENNET_GRES:?}"
MAIL="${TAUGENNET_MAIL:-}"
OLD_ROOT="/scratch/network/sz3962/taugennet"; OLD_CONDA_A="/home/sz3962/.conda/envs/taugennet"
OLD_CONDA_B="/scratch/network/sz3962/.conda/envs/taugennet"; OLD_MAIL="sz3962@princeton.edu"
export TAUGENNET_ROOT TAUGENNET_CONDA_ENV TAUGENNET_PARTITION TAUGENNET_GRES MAIL OLD_ROOT OLD_CONDA_A OLD_CONDA_B OLD_MAIL APPLY
total=0; changed=0
while IFS= read -r f; do total=$((total+1))
  n=$(APPLY="$APPLY" perl -e '
    local $/; my $s=<STDIN>; my $o=$s;
    $s=~s/\Q$ENV{OLD_CONDA_A}\E/$ENV{TAUGENNET_CONDA_ENV}/g;
    $s=~s/\Q$ENV{OLD_CONDA_B}\E/$ENV{TAUGENNET_CONDA_ENV}/g;
    $s=~s/\Q$ENV{OLD_ROOT}\E/$ENV{TAUGENNET_ROOT}/g;
    $s=~s/^#SBATCH --partition=\S+/#SBATCH --partition=$ENV{TAUGENNET_PARTITION}/mg;
    $s=~s/^#SBATCH --gres=\S+/#SBATCH --gres=$ENV{TAUGENNET_GRES}/mg;
    if(length $ENV{MAIL}){$s=~s/^#SBATCH --mail-user=\S+/#SBATCH --mail-user=$ENV{MAIL}/mg}
    else{$s=~s/^#SBATCH --mail-(user|type)=\S+\n//mg}
    my $d=($s ne $o)?1:0; if($d && $ENV{APPLY} eq "1"){print STDERR $s} print $d;
  ' < "$f" 2> "$f.new")
  if [ "$n" = "1" ]; then changed=$((changed+1))
    if [ "$APPLY" -eq 1 ]; then cp "$f" "$f.bak"; mv "$f.new" "$f"; else rm -f "$f.new"; fi
  else rm -f "$f.new"; fi
done < <(find "$SLURM_DIR" -name '*.slurm' | sort)
echo "patched $changed of $total slurm scripts"
