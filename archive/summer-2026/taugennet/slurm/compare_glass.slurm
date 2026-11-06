#!/bin/bash
#SBATCH --job-name=cmp_glass
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=results/logs/cmp_glass_%j.out
set -x
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
export TAUGENNET_ROOT=$HOME/taugennet
cd $HOME/taugennet
PY=/usr/bin/python3.9

# arch comparison: small vs large vs guidance (gens already cached)
$PY scripts/compare_glass.py --fold 0 --subjects 0 1 2 \
  --panel c03_w1:results/generated/arch_sweep/cfgm_5/c03/fold_0/w1 \
  --panel c04_w1:results/generated/arch_sweep/cfgm_5/c04/fold_0/w1 \
  --panel c04_w7:results/generated/arch_sweep/cfgm_5/c04/fold_0/w7 \
  --out results/figures/compare/c03_vs_c04.png

# epoch comparison: only runs once the final_c04 jobs have produced gens
E65=results/generated/final/cfgm_5_c04_e65/w1
E520=results/generated/final/cfgm_5_c04_e520/w1
if [ -d "$E65" ] && [ -d "$E520" ]; then
  $PY scripts/compare_glass.py --fold 0 --subjects 0 1 2 \
    --panel e65:$E65 --panel e520:$E520 \
    --out results/figures/compare/e65_vs_e520.png
else
  echo "SKIP e65-vs-e520: final gens not ready yet ($E65 / $E520)"
fi

bash scripts/publish_to_fac.sh
