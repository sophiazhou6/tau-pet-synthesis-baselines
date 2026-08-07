import os, re, sys
import numpy as np, pandas as pd
from denseunet.config import build_dataloaders

GROUP={1:"CN",2:"MCI",3:"AD"}
TAGS={"controls322_plain_v2":("suvr",False),
      "controls322_paint_atrophy":("atrophy",True),
      "controls322_paint_both":("both",True),
      "controls322_suvr_lr1e-4_wd0":("suvr",True)}

sp=os.path.expanduser("~/taugennet/data/raw/controls_322")
grp={}
for f in ("heldout_test_split.csv","train_val_split.csv"):
    d=pd.read_csv(os.path.join(sp,f))
    for _,r in d.iterrows(): grp[str(int(r["RID"]))]=GROUP.get(int(r["group"]),"?")

for tag,(paint,use_suvr) in TAGS.items():
    gd=f"results/generated/{tag}"
    if not os.path.isdir(gd): print(f"SKIP {tag}"); continue
    os.environ["TAUGENNET_PAINT"]=paint
    _,_,te,*_ = build_dataloaders(mode="atrophy", use_dk_mask=True, val_frac=0.0,
                                  use_controls_322=True, suvr_input=use_suvr)
    mask = te._dk_mask.squeeze().numpy() > 0
    rids = [re.search(r"RID[_-](\d+)", p).group(1) for p in te.pet_paths]
    rows=[]
    for i in range(len(te)):
        f=os.path.join(gd, f"subject_{i:03d}.npy")
        if not os.path.exists(f): continue
        pet,_,_ = te[i]                                   # populates te._pet_norms[i]
        lo,hi = te._pet_norms[i]                          # masked min/max, SUVR space
        rng = float(hi) - float(lo)
        real_n = pet.squeeze().numpy()[mask]
        gen_n  = np.load(f).squeeze()[mask]
        real   = real_n*rng + float(lo)                   # -> SUVR
        gen    = gen_n *rng + float(lo)
        mse = float(np.mean((real-gen)**2))
        rows.append({"subject_idx":f"subject_{i:03d}", "RID":rids[i],
                     "group":grp.get(rids[i],"?"),
                     "pearson_wholebrain":float(np.corrcoef(real,gen)[0,1]),
                     "mse_wholebrain":mse,
                     "rmse_wholebrain":float(np.sqrt(mse)),
                     "mae_wholebrain":float(np.mean(np.abs(real-gen)))})
    df=pd.DataFrame(rows)
    out=f"results/records/denseunet_baseline/{tag}/per_subject_wholebrain.csv"
    df.to_csv(out,index=False)
    print(f"{tag:<34} n={len(df):<3} r={df.pearson_wholebrain.mean():.4f}±{df.pearson_wholebrain.std():.4f} "
          f"mse={df.mse_wholebrain.mean():.4f}±{df.mse_wholebrain.std():.4f} "
          f"mae={df.mae_wholebrain.mean():.4f}±{df.mae_wholebrain.std():.4f}")
