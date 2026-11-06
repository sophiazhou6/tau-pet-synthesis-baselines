# Null benchmark (average training scan as the prediction) + is any subject-specific pattern left once scale is gone?
import os, re, sys, numpy as np, pandas as pd, torch
GH="/scratch/network/sz3962/tau-pet-synthesis-baselines/taugennet"; sys.path.insert(0,GH)
os.environ["TAUGENNET_ROOT"]=GH
from src.dataset_final import build_dataloaders
torch.set_num_threads(4)
tr,_,te,*_=build_dataloaders("atrophy",use_controls_322=True,use_dk_mask=True,train_frac=0.8,val_frac=0.0)
mt=te._dk_mask; mask=mt.squeeze().numpy()>0
rid=lambda p:int(re.search(r"RID[_-](\d+)",p).group(1))
def load(ds,keep=True):
    o={}
    for p in ds.pet_paths:
        v,lo,hi=ds._load(p,mt); o[rid(p)]=(float(lo),float(hi),(v.squeeze().numpy()[mask]*(hi-lo)+lo).astype(np.float32))
    return o
P=load(tr); T=load(te); C=np.mean([v[1] for v in P.values()])
tmpl=np.mean([v[2] for v in P.values()],axis=0)
tn=np.mean([v[2]/v[1] for v in P.values()],axis=0)   # average of [0,1]-scaled training scans
rs=[np.corrcoef(v[2],tmpl)[0,1] for v in T.values()]
real=np.concatenate([v[2] for v in T.values()]); g=np.concatenate([tmpl]*len(T)); gown=np.concatenate([tn*v[1] for v in T.values()])
print(f"AVERAGE TRAINING SCAN as the prediction, test n={len(T)}: per-subject r {np.mean(rs):.3f}±{np.std(rs,ddof=1):.3f}; pooled r (no real scale) {np.corrcoef(real,g)[0,1]:.3f}; MSE {np.mean((real-g)**2):.3f}")
print(f"  same average, scaled by each subject's REAL max (as our scorer does): pooled r {np.corrcoef(real,gown)[0,1]:.3f}; MSE {np.mean((real-gown)**2):.3f}")
GEN="/scratch/network/sz3962/tau-denseunet-baseline/results/generated/baselines/cohort_v2"
REC="/scratch/network/sz3962/tau-denseunet-baseline/results/records/baselines/cohort_v2"
print("\nrun | per-subj r | r vs the average scan's own per-subject r (paired) | relative-pattern SET3 | corr(file max, real max)")
for tag,sub in [("denseunet_paper_refit","single"),("taugennet_paper","single"),("taugennet_paper","fold_0"),("taugennet_paper","fold_2"),("taugennet_paper","fold_3"),("taugennet_paper","fold_4")]:
    z=np.load(f"{GEN}/{tag}/{sub}/predictions.npz"); idx=pd.read_csv(f"{GEN}/{tag}/{sub}/index.csv"); rids=[int(x) for x in idx.RID]
    gn=[z[f"subject_{i:03d}"].squeeze()[mask] for i in range(len(rids))]
    hi=np.array([T[r][1] for r in rids]); lo=np.array([T[r][0] for r in rids])
    ps=np.array([np.corrcoef(a,T[r][2])[0,1] for a,r in zip(gn,rids)]); pt=np.array([np.corrcoef(tmpl,T[r][2])[0,1] for r in rids])
    Rg=pd.read_csv(f"{REC}/{tag}/{sub}/regional_means_gen_atrophy.csv",index_col=0).select_dtypes("number").values
    Rr=pd.read_csv(f"{REC}/{tag}/{sub}/regional_means_real_atrophy.csv",index_col=0).select_dtypes("number").values
    Rn=(Rg-lo[:,None])/(hi-lo+1e-8)[:,None]
    Gp=Rn/Rn.mean(1,keepdims=True); Rp=Rr/Rr.mean(1,keepdims=True)
    rel=np.nanmean([np.corrcoef(Gp[:,k],Rp[:,k])[0,1] for k in range(86)])
    fm=np.array([(a*(h-l)+l).max() for a,h,l in zip(gn,hi,lo)])
    print(f"{tag}/{sub:8s} r {ps.mean():.3f} | avg-scan r {pt.mean():.3f}, model higher in {(ps>pt).sum()}/{len(ps)}, mean diff {np.mean(ps-pt):+.3f} | rel-pattern SET3 {rel:.3f} | corr(file max, real max) {np.corrcoef(fm,hi)[0,1]:.3f}, file max/real max mean {np.mean(fm/hi):.3f} sd {np.std(fm/hi):.3f}")
