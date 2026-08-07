import os, re
import numpy as np, pandas as pd
from src import dataset_spatial as DS

GROUP={1:"CN",2:"MCI",3:"AD"}
CFG={"final_t1_atrophy_enc":("atrophy","none"),
     "final_t2_suvr_enc":("suvr","none"),
     "final_t2b_both_enc":("both","none"),
     "final_t3_suvr_enc_plasma":("suvr","ptau217")}

sp=os.path.expanduser("~/taugennet/data/raw/controls_322")
grp={}
for f in ("heldout_test_split.csv","train_val_split.csv"):
    d=pd.read_csv(os.path.join(sp,f))
    for _,r in d.iterrows(): grp[str(int(r["RID"]))]=GROUP.get(int(r["group"]),"?")

for tag,(mp,cm) in CFG.items():
    gd=f"results/generated/controls_322_TEST/{tag}"
    if not os.path.isdir(gd): print(f"SKIP {tag}"); continue
    os.environ["TAUGENNET_SPATIAL_MAP"]=mp
    _,_,te,*_ = DS.build_dataloaders(use_dk_mask=True, cond_mode=cm, use_controls_322=True)
    mask = te._dk_mask.squeeze().numpy() > 0
    rids = [re.search(r"RID[_-](\d+)", p).group(1) for p in te.pet_paths]
    has_norms = hasattr(te, "get_pet_norms")
    rows=[]
    for i in range(len(te)):
        f=os.path.join(gd, f"subject_{i:03d}.npy")
        if not os.path.exists(f): continue
        pet = te[i][0]
        real_n = pet.squeeze().numpy()[mask]; gen_n = np.load(f).squeeze()[mask]
        rec={"subject_idx":f"subject_{i:03d}","RID":rids[i],"group":grp.get(rids[i],"?"),
             "pearson_wholebrain":float(np.corrcoef(real_n,gen_n)[0,1])}
        if has_norms:
            lo,hi = te.get_pet_norms(i); rng=float(hi)-float(lo)
            real=real_n*rng+float(lo); gen=gen_n*rng+float(lo)
            mse=float(np.mean((real-gen)**2))
            rec.update(mse_wholebrain=mse, rmse_wholebrain=float(np.sqrt(mse)),
                       mae_wholebrain=float(np.mean(np.abs(real-gen))))
        rows.append(rec)
    df=pd.DataFrame(rows)
    out=f"results/records/controls_322_TEST/{tag}/per_subject_wholebrain.csv"
    df.to_csv(out,index=False)
    print(f"{tag:<28} n={len(df):<3} r={df.pearson_wholebrain.mean():.4f}+/-{df.pearson_wholebrain.std():.4f}"
          f"  suvr_metrics={'yes' if has_norms else 'NO'}  groups={dict(df.group.value_counts())}")
