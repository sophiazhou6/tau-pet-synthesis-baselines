# How much of the SUVR-space scores / classifier signal comes from scaling each prediction with the
# subject's REAL PET min/max? Re-scores saved [0,1] predictions with one fixed training-pool scale.
import os, re, sys, json, numpy as np, pandas as pd, torch
GH="/scratch/network/sz3962/tau-pet-synthesis-baselines/taugennet"; sys.path.insert(0,GH)
os.environ["TAUGENNET_ROOT"]=GH
from src.dataset_final import build_dataloaders
from sklearn.metrics import roc_auc_score
torch.set_num_threads(4)
tr,_,te,*_=build_dataloaders("atrophy",use_controls_322=True,use_dk_mask=True,train_frac=0.8,val_frac=0.0)
mask_t=te._dk_mask; mask=mask_t.squeeze().numpy()>0
print("pool",len(tr),"test",len(te),"mask voxels",mask.sum(),flush=True)
rid=lambda p:int(re.search(r"RID[_-](\d+)",p).group(1))
def load(ds):
    out={}
    for p in ds.pet_paths:
        v,lo,hi=ds._load(p,mask_t)
        out[rid(p)]=(float(lo),float(hi),(v.squeeze().numpy()[mask]*(hi-lo)+lo).astype(np.float32))
    return out
T=load(te); P=load(tr)
pet,_,_=te[0]; lo0,hi0=te._pet_norms[0]; r0=rid(te.pet_paths[0])
print(f"loader check RID {r0}: dataset lo/hi {float(lo0):.4f}/{float(hi0):.4f}  mine {T[r0][0]:.4f}/{T[r0][1]:.4f}",flush=True)
A={**P,**T}
hi_tr=np.array([v[1] for v in P.values()]); lo_tr=np.array([v[0] for v in P.values()])
C_hi,C_lo=hi_tr.mean(),lo_tr.mean()
hi_te=np.array([v[1] for v in T.values()])
print(f"real max SUVR: pool mean {C_hi:.3f} sd {hi_tr.std(ddof=1):.3f} range {hi_tr.min():.2f}-{hi_tr.max():.2f}; test mean {hi_te.mean():.3f} sd {hi_te.std(ddof=1):.3f}; min: pool mean {C_lo:.4f}",flush=True)
GEN="/scratch/network/sz3962/tau-denseunet-baseline/results/generated/baselines/cohort_v2"
REC="/scratch/network/sz3962/tau-denseunet-baseline/results/records/baselines/cohort_v2"
def pool_r(r,g): return float(np.corrcoef(np.concatenate(r),np.concatenate(g))[0,1])
rows=[]; aucs=[]
def run(tag,sub):
    z=np.load(f"{GEN}/{tag}/{sub}/predictions.npz"); idx=pd.read_csv(f"{GEN}/{tag}/{sub}/index.csv")
    rids=[int(x) for x in idx.RID]
    gn=[z[f"subject_{i:03d}"].squeeze()[mask] for i in range(len(rids))]
    real=[A[r][2] for r in rids]; lo=np.array([A[r][0] for r in rids]); hi=np.array([A[r][1] for r in rids])
    own=[g*(h-l)+l for g,h,l in zip(gn,hi,lo)]; fix=[g*(C_hi-C_lo)+C_lo for g in gn]
    Rg=pd.read_csv(f"{REC}/{tag}/{sub}/regional_means_gen_atrophy.csv",index_col=0).select_dtypes("number").values
    Rr=pd.read_csv(f"{REC}/{tag}/{sub}/regional_means_real_atrophy.csv",index_col=0).select_dtypes("number").values
    Rn=(Rg-lo[:,None])/(hi-lo+1e-8)[:,None]; Rf=Rn*(C_hi-C_lo)+C_lo
    s3=lambda G:float(np.nanmean([np.corrcoef(G[:,k],Rr[:,k])[0,1] for k in range(Rr.shape[1])]))
    mse=lambda g:float(np.mean(np.concatenate([(a-b)**2 for a,b in zip(g,real)])))
    loc=json.load(open(f"{REC}/{tag}/{sub}/localization.json"))
    rows.append(dict(run=f"{tag}/{sub}",n=len(rids),SET1_own=pool_r(real,own),SET1_fixed=pool_r(real,fix),SET3_own=s3(Rg),SET3_scored=loc["set3"],SET3_fixed=s3(Rf),MSE_own=mse(own),MSE_fixed=mse(fix),
       brightness_r_own=float(np.corrcoef(Rg.mean(1),Rr.mean(1))[0,1]),brightness_r_fixed=float(np.corrcoef(Rn.mean(1),Rr.mean(1))[0,1]),max_vs_real_mean_r=float(np.corrcoef(hi,Rr.mean(1))[0,1])))
    if sub in("single","fold_0","fold_2","fold_3","fold_4") and "_val" not in sub:
        ps=pd.read_csv(f"{REC}/{tag}/{sub}/per_subject_wholebrain.csv"); grp=ps.group.values
        for name,pos in (("AD vs CN",["AD"]),("MCI+AD vs CN",["MCI","AD"])):
            keep=np.isin(grp,pos+["CN"]); y=np.isin(grp[keep],pos).astype(int)
            aucs.append(dict(run=f"{tag}/{sub}",task=name,n=int(keep.sum()),real_max_alone=roc_auc_score(y,hi[keep]),pred_as_exported=roc_auc_score(y,Rg.mean(1)[keep]),pred_no_real_scale=roc_auc_score(y,Rn.mean(1)[keep]),real_scan_mean=roc_auc_score(y,Rr.mean(1)[keep])))
for t,s in [("denseunet_paper_refit","single"),("taugennet_paper","single")]+[("taugennet_paper",f"fold_{k}") for k in (0,2,3,4)]+[("denseunet_paper",f"fold_{k}") for k in range(5)]+[("denseunet_paper",f"fold_{k}_val") for k in range(5)]+[("taugennet_paper",f"fold_{k}_val") for k in (0,2,3,4)]:
    try: run(t,s); print("done",t,s,flush=True)
    except Exception as e: print("SKIP",t,s,repr(e)[:150],flush=True)
pd.set_option("display.width",230); pd.set_option("display.max_columns",30)
print("\n== scores with the subject's real scale vs one fixed scale"); print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\n== classification signal from a single brain-wide number (AUC)"); print(pd.DataFrame(aucs).round(3).to_string(index=False))
pd.DataFrame(rows).to_csv(os.path.dirname(os.path.abspath(__file__))+"/scale_leak_scores.csv",index=False); pd.DataFrame(aucs).to_csv(os.path.dirname(os.path.abspath(__file__))+"/scale_leak_auc.csv",index=False)
