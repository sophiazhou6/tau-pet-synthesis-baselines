import glob, os, re, sys
import numpy as np, pandas as pd, torch
ROOT = os.path.expanduser("~/tau-denseunet-baseline/results")
# ONLY the controls_322 grid -- never the legacy AD/MCI dirs
cfgs = sorted(glob.glob(os.path.join(ROOT, "records", "denseunet_baseline", "cv322_grid322_*")))
rows = []
for c in cfgs:
    vals = []
    for f in sorted(glob.glob(os.path.join(c, "fold_*"))):
        hit = glob.glob(os.path.join(f, "pooled_wholebrain_*.csv"))
        if hit:
            vals.append(float(pd.read_csv(hit[0], index_col=0)["pearson"].iloc[0]))
    if vals:
        rows.append((os.path.basename(c), np.mean(vals), np.std(vals), len(vals)))
if not rows:
    print("No controls_322 grid records yet under", os.path.join(ROOT,"records","denseunet_baseline")); sys.exit(1)
rows.sort(key=lambda r: -r[1])
print("%-34s %10s %9s %6s" % ("config","SET1 mean","sd","folds"))
for n,m,sd,k in rows:
    print("%-34s %10.4f %9.4f %6d%s" % (n,m,sd,k,"   <-- WINNER" if n==rows[0][0] else ""))
win = rows[0][0]
tag = win.replace("cv322_","")                      # cv322_grid322_lr..._wd... -> grid322_lr..._wd...
eps = []
for f in range(5):
    ck = os.path.join(ROOT,"checkpoints","atrophy",tag,"fold_%d"%f,"denseunet_best.pt")
    if os.path.exists(ck): eps.append(int(torch.load(ck,map_location="cpu")["epoch"]))
med = int(np.median(eps)) if eps else 126
lr = re.search(r"lr([0-9.e+-]+?)_wd", win).group(1)
wd = re.search(r"wd([0-9.e+-]+)$", win).group(1)
print("\nwinner=%s\n  best epochs=%s  median=%d\n  lr=%s  wd=%s" % (win, eps, med, lr, wd))
