import os, re, csv, sys, collections
os.environ["TAUGENNET_SPATIAL_MAP"] = "suvr"
REPO = os.path.expanduser("~/taugennet")
BASE = os.path.join(REPO, "data/raw")
sys.path.insert(0, REPO)
from src.dataset_spatial import build_dataloaders

N_FOLDS = 5
ATLAS = os.path.join(BASE, "mni_dk_atlas.nii.gz")
def rid_group(p):
    m = re.search(r"RID_(\d+)", p)
    g = "AD" if "/AD/" in p else ("MCI" if "/MCI/" in p else "?")
    return (m.group(1) if m else "?"), g

rows = []
for fold in range(N_FOLDS):
    tr, va, te, *_ = build_dataloaders(base_dir=BASE, atlas_path=ATLAS,
                                       cond_mode="ptau217", fold_idx=fold,
                                       n_folds=N_FOLDS, use_mentor_split=True)
    for name, ds in (("train", tr), ("val", va), ("test", te)):
        for p in ds.pet_paths:
            rid, g = rid_group(p)
            rows.append({"RID": rid, "group": g, "fold": fold, "split": name})

out = os.path.join(REPO, "results/records/split_suvr_5fold.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
if os.path.exists(out):
    sys.exit(f"REFUSING TO OVERWRITE {out}")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, ["RID", "group", "fold", "split"]); w.writeheader(); w.writerows(rows)
print("wrote", out, len(rows), "rows")
for fold in range(N_FOLDS):
    tot = collections.Counter(r["split"] for r in rows if r["fold"] == fold)
    byg = collections.Counter((r["split"], r["group"]) for r in rows if r["fold"] == fold)
    print(f"fold {fold}: {dict(tot)}  by group: {dict(byg)}")
