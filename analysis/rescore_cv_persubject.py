import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.environ["TAUGENNET_ROOT"])
from src.dataset_spatial import build_dataloaders
from src.dataset_final import unnormalize

ARM, COND = sys.argv[1], sys.argv[2]
ROOT = os.environ["TAUGENNET_ROOT"]
GEN  = f"{ROOT}/results/generated/controls_322_cv/{ARM}"
OUT  = f"{ROOT}/results/records/controls_322_cv_persubject/{ARM}"
DIAG = {1: "AD", 0: "MCI", 2: "CN"}

print(f"### {ARM}  cond={COND}  MAP={os.environ.get('TAUGENNET_SPATIAL_MAP')} "
      f"REQ_PTAU={os.environ.get('TAUGENNET_REQUIRE_PTAU','-')}")
mismatch = 0
for F in range(5):
    gdir = f"{GEN}/fold_{F}"
    npys = sorted(glob.glob(f"{gdir}/subject_*.npy"))
    if not npys:
        print(f"  fold {F}: NO GENERATIONS, skipping"); continue
    _, val, _, *_ = build_dataloaders(use_controls_322=True, use_dk_mask=True,
                                      cond_mode=COND, fold_idx=F, n_folds=5)
    if len(val) != len(npys):
        print(f"  fold {F}: MISMATCH val={len(val)} npy={len(npys)} — SKIPPING (unsafe)")
        continue
    rows = []
    for i in range(len(val)):
        pet = np.asarray(val[i][0]).squeeze()
        gen = np.load(f"{gdir}/subject_{i:03d}.npy").squeeze()
        m   = pet > 0
        r   = float(np.corrcoef(pet[m], gen[m])[0, 1])          # normalized space
        pmin, pmax = val.get_pet_norms(i)
        rs, gs = unnormalize(pet, pmin, pmax), unnormalize(gen, pmin, pmax)
        d   = rs[m] - gs[m]
        pth = val.pet_paths[i]
        rid = os.path.basename(os.path.dirname(pth)).replace("RID_", "")
        grp = os.path.basename(os.path.dirname(os.path.dirname(pth)))
        mri_grp = DIAG.get(int(val.diagnoses[i]), "?")
        if grp != mri_grp: mismatch += 1
        rows.append(dict(subject_idx=f"subject_{i:03d}", RID=int(rid), group=grp,
                         pearson_wholebrain=r,
                         mse_wholebrain=float((d**2).mean()),
                         rmse_wholebrain=float(np.sqrt((d**2).mean())),
                         mae_wholebrain=float(np.abs(d).mean())))
    od = f"{OUT}/fold_{F}"; os.makedirs(od, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{od}/per_subject_wholebrain.csv", index=False)
    rr = [x["pearson_wholebrain"] for x in rows]
    print(f"  fold {F}: n={len(rows)}  pearson {np.mean(rr):.4f}±{np.std(rr,ddof=1):.4f}  -> {od}")
print(f"  PET-dir vs MRI-dir group disagreements: {mismatch}")
