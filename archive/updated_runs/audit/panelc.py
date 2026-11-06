# Why does panel C's test value sit right of the CV bars? Same models on val vs test subjects; what differs between
# subject sets; how unusual is this test set among random groups drawn from the development pool's out-of-fold predictions.
import os, re, sys, json, numpy as np, pandas as pd, torch
GH="/scratch/network/sz3962/tau-pet-synthesis-baselines/taugennet"; sys.path.insert(0,GH); os.environ["TAUGENNET_ROOT"]=GH
from src.dataset_final import build_dataloaders
torch.set_num_threads(4)
tr,_,te,*_=build_dataloaders("atrophy",use_controls_322=True,use_dk_mask=True,train_frac=0.8,val_frac=0.0)
mt=te._dk_mask
rid=lambda p:int(re.search(r"RID[_-](\d+)",p).group(1))
hi={}
for ds in (tr,te):
    for p in ds.pet_paths:
        _,lo,h=ds._load(p,mt); hi[rid(p)]=float(h)
pd.Series(hi,name="real_max").rename_axis("RID").to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"real_max.csv"))
G="/scratch/network/sz3962/tau-denseunet-baseline/results/generated/baselines/cohort_v2"
R="/scratch/network/sz3962/tau-denseunet-baseline/results/records/baselines/cohort_v2"
def load(tag,sub):
    idx=[int(x) for x in pd.read_csv(f"{G}/{tag}/{sub}/index.csv").RID]
    g=pd.read_csv(f"{R}/{tag}/{sub}/regional_means_gen_atrophy.csv",index_col=0).select_dtypes("number").values
    r=pd.read_csv(f"{R}/{tag}/{sub}/regional_means_real_atrophy.csv",index_col=0).select_dtypes("number").values
    ps=pd.read_csv(f"{R}/{tag}/{sub}/per_subject_wholebrain.csv")
    return np.array(idx),g,r,ps.group.values
def set3(g,r):
    gz=g-g.mean(0); rz=r-r.mean(0)
    return float(np.nanmean((gz*rz).sum(0)/np.sqrt((gz**2).sum(0)*(rz**2).sum(0))))
for tag,final in (("denseunet_paper","denseunet_paper_refit"),("taugennet_paper","taugennet_paper")):
    print(f"\n===== {tag}: fold models on their VAL subjects vs on the same TEST subjects")
    print("fold  n  CN/MCI/AD  SET3_val  SET3_test | real brain mean SD | r(real max, real mean) | largest real max | SET3_val w/o top-1, top-3 real max")
    OG,OR,OH,OGp=[],[],[],[]
    for k in range(5):
        i,g,r,gp=load(tag,f"fold_{k}_val"); _,gt,rt,_=load(tag,f"fold_{k}")
        h=np.array([hi[x] for x in i]); m=r.mean(1); o=np.argsort(-h)
        d1=np.ones(len(i),bool); d1[o[:1]]=False; d3=np.ones(len(i),bool); d3[o[:3]]=False
        print(f"{k:3d} {len(i):3d}  {(gp=='CN').sum():2d}/{(gp=='MCI').sum():2d}/{(gp=='AD').sum():2d}   {set3(g,r):.3f}    {set3(gt,rt):.3f}    | {m.std(ddof=1):.3f} | {np.corrcoef(h,m)[0,1]:.3f} | {h.max():.2f} | {set3(g[d1],r[d1]):.3f}, {set3(g[d3],r[d3]):.3f}")
        OG.append(g);OR.append(r);OH.append(h);OGp.append(gp)
    OG,OR,OH,OGp=map(np.concatenate,(OG,OR,OH,OGp))
    i,g,r,gp=load(final,"single"); h=np.array([hi[x] for x in i]); m=r.mean(1); o=np.argsort(-h); d3=np.ones(len(i),bool); d3[o[:3]]=False
    t=set3(g,r)
    print(f"final model TEST: n={len(i)} {(gp=='CN').sum()}/{(gp=='MCI').sum()}/{(gp=='AD').sum()} SET3 {t:.3f} | {m.std(ddof=1):.3f} | {np.corrcoef(h,m)[0,1]:.3f} | {h.max():.2f} | w/o top-3 {set3(g[d3],r[d3]):.3f}")
    print(f"all 477 out-of-fold together: {(OGp=='CN').sum()}/{(OGp=='MCI').sum()}/{(OGp=='AD').sum()} SET3 {set3(OG,OR):.3f}; r(real max, real mean) {np.corrcoef(OH,OR.mean(1))[0,1]:.3f}; real mean SD {OR.mean(1).std(ddof=1):.3f}")
    rng=np.random.default_rng(0); N=len(OG)
    for n,lab in ((96,"random 96"),(120,"random 120")):
        v=np.array([set3(OG[s],OR[s]) for s in (rng.choice(N,n,replace=False) for _ in range(2000))])
        print(f"{lab} from pool: mean {v.mean():.3f} sd {v.std(ddof=1):.3f} 2.5-97.5% {np.percentile(v,2.5):.3f}..{np.percentile(v,97.5):.3f}; test {t:.3f} is at percentile {(v<t).mean()*100:.1f}")
    idx={x:np.where(OGp==x)[0] for x in ("CN","MCI","AD")}
    mix=[(gp==x).sum() for x in ("CN","MCI","AD")]
    v=np.array([set3(OG[s],OR[s]) for s in (np.concatenate([rng.choice(idx[x],c,replace=False) for x,c in zip(("CN","MCI","AD"),mix)]) for _ in range(2000))])
    print(f"random 120 with test's diagnosis mix: mean {v.mean():.3f} sd {v.std(ddof=1):.3f}; test at percentile {(v<t).mean()*100:.1f}")
    b=np.array([set3(g[s],r[s]) for s in (rng.choice(len(g),96,replace=False) for _ in range(2000))])
    print(f"random 96 from TEST itself: mean {b.mean():.3f} sd {b.std(ddof=1):.3f} 2.5-97.5% {np.percentile(b,2.5):.3f}..{np.percentile(b,97.5):.3f}")
