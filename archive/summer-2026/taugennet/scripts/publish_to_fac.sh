#!/bin/bash
set -u
FAC_DB="${FAC_DB:-$(find /mnt/fac -maxdepth 6 -type d -iname 'Diffusion_Baseline' 2>/dev/null | head -1)}"
for i in $(seq 1 12); do
  [ -n "$FAC_DB" ] && ls "$FAC_DB" >/dev/null 2>&1 && break
  FAC_DB="$(find /mnt/fac -maxdepth 6 -type d -iname 'Diffusion_Baseline' 2>/dev/null | head -1)"; sleep 5
done
if [ -z "$FAC_DB" ] || ! ls "$FAC_DB" >/dev/null 2>&1; then
  echo "[publish_to_fac] FAC unreachable — outputs stay in local results/"; exit 0; fi
echo "[publish_to_fac] FAC_DB=$FAC_DB"
for T in results/figures results/records; do
  [ -e "$T" ] || continue
  mkdir -p "$FAC_DB/results"
  rsync -a "$T" "$FAC_DB/results/" && echo "[publish_to_fac] synced $T -> $FAC_DB/results/"
done
echo "[publish_to_fac] done."
