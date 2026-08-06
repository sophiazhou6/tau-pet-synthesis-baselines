#!/bin/bash
# Re-runnable: rebuild every table + figure from whatever has finished, push to FAC.
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
export TAUGENNET_ROOT=$HOME/taugennet
cd $HOME/taugennet
PY=/usr/bin/python3.9
OUT=results/records/final_report
mkdir -p $OUT results/figures/compare

echo "##### 1. TABLES #####"
$PY - <<'PYEOF' 2>&1 | tee $OUT/REPORT.txt
import os, numpy as np, pandas as pd
ROOT='results/records/arch_sweep/cfgm_5'
OUT='results/records/final_report'
ARCH={'c01':'lch2 small','c02':'lch2 LARGE','c03':'lch3 small','c04':'lch3 LARGE','c05':'lch4 small',
      'c06':'lch4 LARGE','c07':'lch5 small','c08':'lch5 LARGE','c09':'lch6 small','c10':'lch6 LARGE',
      'c11':'lch3 KL1e-3','c12':'lch3 KL1e-6','c13':'lch3 small COSINE'}
SKIP=('label','region','index','unnamed')
def read(base):
    out={}
    for tag, fn, agg in (('S1','pooled_wholebrain_atrophy.csv','first'),
                         ('S2','region_pooled_atrophy.csv','mean'),
                         ('S3','region_crosssubject_metrics_atrophy.csv','mean')):
        p=os.path.join(base, fn)
        if not os.path.exists(p): continue
        d=pd.read_csv(p)
        for c in d.columns:
            if d[c].dtype.kind in 'fi' and not any(s in c.lower() for s in SKIP):
                out[f'{tag}_{c}'] = float(d[c].iloc[0]) if agg=='first' else float(d[c].mean())
    return out or None

rows=[]
for cfg in sorted(os.listdir(ROOT)):
    for f in range(5):
        for w in (1,5,7,10,15,20):
            r=read(f'{ROOT}/{cfg}/fold_{f}/w{w}')
            if r:
                r.update(cfg=cfg, arch=ARCH.get(cfg,cfg), fold=f, w=w); rows.append(r)
pf=pd.DataFrame(rows)
pf.to_csv(f'{OUT}/perfold_all.csv', index=False)
mcols=[c for c in pf.columns if c.startswith(('S1_','S2_','S3_'))]

print('='*90); print('SWEEP: CV mean over folds, w=1  (n = folds completed)'); print('='*90)
a=pf[pf.w==1].groupby(['cfg','arch']).agg(n=('fold','count'), **{c:(c,'mean') for c in mcols}).reset_index()
a=a.sort_values('S3_pearson', ascending=False)
a.to_csv(f'{OUT}/sweep_cvmean_w1.csv', index=False)
print(a.to_string(index=False, float_format='%.4f'))

print(); print('='*90); print('LIKE-FOR-LIKE: only folds every listed config has finished'); print('='*90)
w1=pf[pf.w==1]
common=set(range(5))
for cfg in w1.cfg.unique(): common &= set(w1[w1.cfg==cfg].fold)
if common:
    print('common folds:', sorted(common))
    b=w1[w1.fold.isin(common)].groupby(['cfg','arch']).agg(n=('fold','count'), **{c:(c,'mean') for c in mcols}).reset_index()
    b=b.sort_values('S3_pearson', ascending=False); b.to_csv(f'{OUT}/sweep_like_for_like.csv', index=False)
    print(b.to_string(index=False, float_format='%.4f'))
else:
    print('(no fold common to every config)')

print(); print('='*90); print('GUIDANCE SWEEP: S3_pearson by weight'); print('='*90)
g=pf.pivot_table(index=['cfg','arch'], columns='w', values='S3_pearson')
g.to_csv(f'{OUT}/guidance_sweep_S3.csv')
print(g.to_string(float_format='%.3f'))

print(); print('='*90); print('PER-FOLD SPREAD, w=1'); print('='*90)
s=pf[pf.w==1].groupby(['cfg','arch'])['S3_pearson'].agg(['count','mean','std','min','max']).reset_index()
print(s.sort_values('mean', ascending=False).to_string(index=False, float_format='%.4f'))

print(); print('='*90); print('PHASE 6 — final models (all-dev, no val split)'); print('='*90)
fr=[]
for cfg in ('c04','c13'):
    for E in ('e65','e520'):
        r=read(f'results/records/final/cfgm_5_{cfg}_{E}/w1')
        if r: r.update(model=cfg, epochs=E); fr.append(r)
if fr:
    fd=pd.DataFrame(fr); fd.to_csv(f'{OUT}/phase6.csv', index=False)
    front=['model','epochs','S1_pearson','S3_pearson','S1_ssim','S1_mse','S3_mse']
    print(fd[[c for c in front if c in fd.columns]].to_string(index=False, float_format='%.4f'))
    print('\nCompare against the CV mean (c04 S3~0.466). Much lower => the run is broken, do not present it.')
else:
    print('(no Phase 6 results yet)')
PYEOF

echo "##### 2. FIGURES #####"
run(){ $PY scripts/compare_glass.py "$@" || echo "  skip: $*"; }
run --fold 0 --subjects 0 1 2 \
  --panel c03_w1:results/generated/arch_sweep/cfgm_5/c03/fold_0/w1 \
  --panel c04_w1:results/generated/arch_sweep/cfgm_5/c04/fold_0/w1 \
  --panel c04_w7:results/generated/arch_sweep/cfgm_5/c04/fold_0/w7 \
  --out results/figures/compare/c03_vs_c04.png
run --fold 0 --subjects 0 1 2 \
  --panel c03_linear:results/generated/arch_sweep/cfgm_5/c03/fold_0/w1 \
  --panel c13_cosine:results/generated/arch_sweep/cfgm_5/c13/fold_0/w1 \
  --panel c04_large:results/generated/arch_sweep/cfgm_5/c04/fold_0/w1 \
  --out results/figures/compare/schedule_effect.png
run --fold 0 --subjects 0 1 2 3 4 5 \
  --panel c04_w1:results/generated/arch_sweep/cfgm_5/c04/fold_0/w1 \
  --out results/figures/compare/c04_six_subjects.png
run --fold 0 --subjects 0 1 2 --mode slice \
  --panel c03_linear:results/generated/arch_sweep/cfgm_5/c03/fold_0/w1 \
  --panel c13_cosine:results/generated/arch_sweep/cfgm_5/c13/fold_0/w1 \
  --out results/figures/compare/schedule_effect_slices.png
for M in c04 c13; do
  run --fold 0 --subjects 0 1 2 \
    --panel ${M}_e65:results/generated/final/cfgm_5_${M}_e65/w1 \
    --panel ${M}_e520:results/generated/final/cfgm_5_${M}_e520/w1 \
    --out results/figures/compare/${M}_e65_vs_e520.png
done

echo "##### 3. PUBLISH #####"
bash scripts/publish_to_fac.sh
echo
echo "On FAC under Diffusion_Baseline/results/:"
echo "  records/final_report/REPORT.txt        <- read this first"
echo "  records/final_report/*.csv             <- every number, for offline analysis"
echo "  records/SWEEP_SUMMARY.md               <- the writeup"
echo "  figures/compare/*.png                  <- all comparison figures"
