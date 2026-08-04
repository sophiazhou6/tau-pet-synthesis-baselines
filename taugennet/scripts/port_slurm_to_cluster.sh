#!/bin/bash
# port_slurm_to_cluster.sh — rewrite all slurm/*.slurm headers for a new cluster.
#
# SBATCH directives are parsed by SLURM as comments BEFORE the shell runs, so they
# cannot read $TAUGENNET_* env vars — the literal Princeton values must be replaced
# in-file. This script does that from the values in cluster.env.
#
#   source cluster.env
#   bash scripts/port_slurm_to_cluster.sh           # DRY RUN: show what would change
#   bash scripts/port_slurm_to_cluster.sh --apply   # write changes (.bak backups)
#
# Idempotent-ish: matches the OLD Princeton literals, so re-running after --apply
# is a no-op. Restore anytime with:  find slurm -name '*.bak' -exec sh -c 'mv "$1" "${1%.bak}"' _ {} \;

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SLURM_DIR="$ROOT_DIR/slurm"
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1

# ---- required values (from cluster.env) ------------------------------------
: "${TAUGENNET_ROOT:?source cluster.env first (TAUGENNET_ROOT unset)}"
: "${TAUGENNET_CONDA_ENV:?source cluster.env first (TAUGENNET_CONDA_ENV unset)}"
: "${TAUGENNET_PARTITION:?source cluster.env first (TAUGENNET_PARTITION unset)}"
: "${TAUGENNET_GRES:?source cluster.env first (TAUGENNET_GRES unset)}"
MAIL="${TAUGENNET_MAIL:-}"

# OLD literals present in the bundle (from the audit):
OLD_ROOT="/scratch/network/sz3962/taugennet"
OLD_CONDA_A="/home/sz3962/.conda/envs/taugennet"
OLD_CONDA_B="/scratch/network/sz3962/.conda/envs/taugennet"
OLD_MAIL="sz3962@princeton.edu"

echo "Porting slurm headers under: $SLURM_DIR"
echo "  repo root  : $OLD_ROOT           -> $TAUGENNET_ROOT"
echo "  conda env  : $OLD_CONDA_A (+scratch) -> $TAUGENNET_CONDA_ENV"
echo "  partition  : gpu|all              -> $TAUGENNET_PARTITION"
echo "  gres       : a100/gpu:1/mig       -> $TAUGENNET_GRES"
echo "  mail-user  : $OLD_MAIL -> ${MAIL:-<stripped>}"
[ "$APPLY" -eq 1 ] && echo "MODE: APPLY (writing .bak backups)" || echo "MODE: DRY RUN (no files changed) — add --apply to write"
echo

export TAUGENNET_ROOT TAUGENNET_CONDA_ENV TAUGENNET_PARTITION TAUGENNET_GRES MAIL
export OLD_ROOT OLD_CONDA_A OLD_CONDA_B OLD_MAIL APPLY

total=0; changed=0
while IFS= read -r f; do
  total=$((total+1))
  # Count / preview hits, then (optionally) rewrite, all in one perl pass.
  n=$(APPLY="$APPLY" perl -e '
    local $/; my $s = <STDIN>; my $orig = $s;
    my ($root,$ca,$cb,$part,$gres,$mail) =
      ($ENV{TAUGENNET_ROOT},$ENV{TAUGENNET_CONDA_ENV},$ENV{TAUGENNET_CONDA_ENV},
       $ENV{TAUGENNET_PARTITION},$ENV{TAUGENNET_GRES},$ENV{MAIL});
    # conda env prefixes first (both bin/python3 and bin/activate resolve)
    $s =~ s/\Q$ENV{OLD_CONDA_A}\E/$ca/g;
    $s =~ s/\Q$ENV{OLD_CONDA_B}\E/$cb/g;
    $s =~ s/\Q$ENV{OLD_ROOT}\E/$root/g;
    $s =~ s/^#SBATCH --partition=\S+/#SBATCH --partition=$part/mg;
    $s =~ s/^#SBATCH --gres=\S+/#SBATCH --gres=$gres/mg;
    if (length $mail) { $s =~ s/^#SBATCH --mail-user=\S+/#SBATCH --mail-user=$mail/mg; }
    else { $s =~ s/^#SBATCH --mail-(user|type)=\S+\n//mg; }
    my $diff = ($s ne $orig) ? 1 : 0;
    if ($diff && $ENV{APPLY} eq "1") { print STDERR $s; }
    print $diff;
  ' < "$f" 2> "$f.new")
  if [ "$n" = "1" ]; then
    changed=$((changed+1))
    if [ "$APPLY" -eq 1 ]; then
      cp "$f" "$f.bak"; mv "$f.new" "$f"
    else
      rm -f "$f.new"
    fi
    echo "  changed: ${f#$ROOT_DIR/}"
  else
    rm -f "$f.new"
  fi
done < <(find "$SLURM_DIR" -name '*.slurm' | sort)

echo
echo "Summary: $changed of $total slurm scripts would change."
[ "$APPLY" -eq 1 ] && echo "Done. Backups saved as *.bak." || echo "Dry run only. Re-run with --apply to write."

# Reminder for the .sh launchers (not .slurm), which also embed paths:
echo
echo "NOTE: also review the shell launchers (not auto-patched):"
grep -rl "$OLD_ROOT" "$SLURM_DIR" --include='*.sh' 2>/dev/null | sed "s#^#  #" || true
