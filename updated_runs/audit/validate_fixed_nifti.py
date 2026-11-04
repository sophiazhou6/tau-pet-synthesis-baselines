# Reads the exported fixed-scale NIfTIs back and checks them against the saved predictions, the real scans and the audit.
import os, re, sys, numpy as np, pandas as pd, nibabel as nib
GH="/scratch/network/sz3962/tau-pet-synthesis-baselines/taugennet"; os.environ["TAUGENNET_ROOT"]=GH; sys.path[:0]=[GH,GH+"/scripts"]
import evaluate_final as ef
from src.dataset_final import build_dataloaders
from sklearn.metrics import roc_auc_score
_,_,te,*_=build_dataloaders("atrophy",use_controls_322=True,use_dk_mask=True)
mt=te._dk_mask; mask=mt.squeeze().numpy()>0; labels=ef._dk86_label_volume(te)
rids=[int(re.search(r"RID[_-](\d+)",p).group(1)) for p in te.pet_paths]
real,hi=[],[]
for p in te.pet_paths:
    v,lo,h=te._load(p,mt); real.append(v.squeeze().numpy()*(h-lo)+lo); hi.append(h)
hi=np.array(hi); rmean=np.array([r[mask].mean() for r in real])
R="/scratch/network/sz3962/tau-denseunet-baseline/results"
ref=None
for tag,old in (("denseunet_paper_refit",True),("taugennet_paper",False)):
    d=f"{R}/export_nifti/cohort_v2/{tag}/single/test_fixed_scale"; idx=pd.read_csv(f"{d}/index.csv"); G=f"{R}/generated/baselines/cohort_v2/{tag}/single"
    z=np.load(f"{G}/predictions.npz"); lo,up=idx.scale_min[0],idx.scale_max[0]
    assert [int(x) for x in idx.RID]==rids, "order"
    vols,aff=[],[]; dmax=0
    for i,row in enumerate(idx.itertuples()):
        im=nib.load(f"{d}/{row.file}"); v=im.get_fdata().astype(np.float32); vols.append(v); aff.append(im.affine)
        dmax=max(dmax,float(np.abs(v-(z[f"subject_{i:03d}"].squeeze()*(up-lo+1e-8)+lo))[mask].max()))
    if ref is None: ref=aff[0]
    masked=[v*mask for v in vols]
    fm=np.array([v[mask].max() for v in vols]); bm=np.array([v[mask].mean() for v in vols])
    out=[np.abs(v[~mask]) for v in vols]
    r_all=np.concatenate([r[mask] for r in real]); g_all=np.concatenate([v[mask] for v in vols])
    set3,_,_=ef.compute_region_crosssubject_metrics(real,masked,labels)
    ps=np.array([np.corrcoef(r[mask],v[mask])[0,1] for r,v in zip(real,vols)]); sc=pd.read_csv(f"{R}/records/baselines/cohort_v2/{tag}/single/per_subject_wholebrain.csv")
    gp=idx.group.values; keep=np.isin(gp,["AD","CN"]); y=(gp[keep]=="AD").astype(int)
    print(f"\n== {tag}: {len(vols)} files, shape {vols[0].shape}, files whose affine differs from the first file by >1e-3: {sum(not np.allclose(a,ref,atol=1e-3) for a in aff)}, scale {lo:.5f}..{up:.5f}, max in-brain diff from prediction x scale {dmax:.2e}")
    print(f"   corr(file max, real max) {np.corrcoef(fm,hi)[0,1]:+.3f}; corr(file brain mean, real brain mean) {np.corrcoef(bm,rmean)[0,1]:+.3f}")
    print(f"   SET1 {np.corrcoef(r_all,g_all)[0,1]:.3f}  SET3 {float(np.nanmean(set3['pearson'])):.3f}  MSE {np.mean((r_all-g_all)**2):.3f}; per-subject r {ps.mean():.3f} (scored {sc.pearson_wholebrain.mean():.3f}, max diff {np.abs(ps-sc.pearson_wholebrain.values).max():.1e})")
    print(f"   outside the brain mask: mean |value| {np.mean([o.mean() for o in out]):.4f}, largest {max(o.max() for o in out):.3f}, voxels nonzero {100*np.mean([(o>0).mean() for o in out]):.1f}%")
    print("   brain mean by group, file vs real:",{g:(round(float(bm[gp==g].mean()),3),round(float(rmean[gp==g].mean()),3)) for g in ("CN","MCI","AD")},f" AD vs CN AUC file mean {roc_auc_score(y,bm[keep]):.3f}, real {roc_auc_score(y,rmean[keep]):.3f}")
    if "plasma_imputed" in idx: print("   plasma imputed in test:",int(idx.plasma_imputed.sum()),"of",len(idx))
