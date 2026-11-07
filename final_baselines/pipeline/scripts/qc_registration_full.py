import glob, os, sys
import numpy as np, nibabel as nib, torch, torch.nn.functional as F
import pandas as pd
sys.path.insert(0, "/scratch/network/sz3962/taugennet")
from src.atlas_labels import load_atlas_on_scan_grid
MY = "/scratch/network/sz3962/taugennet/data/mni_t1"; VOL = (96, 112, 96)
def loadv(p):
    v = nib.load(p).get_fdata().astype(np.float32)
    return F.interpolate(torch.from_numpy(v)[None, None], size=VOL, mode="trilinear", align_corners=False)[0, 0].numpy()
atlas = load_atlas_on_scan_grid()  # header-aware; qc_full.csv (Oct 2026) predates this and used the mirrored atlas
mask = F.interpolate(torch.from_numpy((atlas > 0).astype(np.float32))[None, None], size=VOL, mode="nearest")[0, 0].numpy() > 0
tpl = loadv("/scratch/network/sz3962/.conda/envs/fslreg/data/standard/MNI152_T1_2mm.nii.gz")
gt = np.linalg.norm(np.stack(np.gradient(tpl)), axis=0)
rows = []
for f in sorted(glob.glob(f"{MY}/*/T1_to_MNI_nonlin.nii.gz")):
    try:
        v = loadv(f)
        ge = np.linalg.norm(np.stack(np.gradient(v)), axis=0)
        rows.append(dict(PTID=os.path.basename(os.path.dirname(f)),
                         r=float(np.corrcoef(v[mask], tpl[mask])[0, 1]),
                         r_edge=float(np.corrcoef(ge[mask], gt[mask])[0, 1]),
                         tissue=float((v[mask] > 0.15 * np.percentile(v, 99.5)).mean())))
    except Exception as e:
        rows.append(dict(PTID=os.path.basename(os.path.dirname(f)), r=np.nan, r_edge=np.nan, tissue=0.0))
q = pd.DataFrame(rows)
pin = pd.read_csv("/scratch/network/sz3962/tau-denseunet-baseline/results/for_anil/pinned_scans_597.csv", dtype={"RID": str})
q = q.merge(pin[["PTID", "RID", "split", "group", "mri_file", "box_dir"]], on="PTID", how="left")
q["pass"] = (q.r >= 0.35) & (q.r_edge >= 0.30) & (q.tissue >= 0.85)
q.to_csv(f"{MY}/qc_full.csv", index=False)
print(f"QC over {len(q)}: pass {int(q['pass'].sum())}, fail {int((~q['pass']).sum())} (held-out fails: {int((~q['pass'] & (q.split == 'test')).sum())})")
print(q[~q["pass"]].sort_values("r")[["PTID", "RID", "split", "group", "r", "r_edge", "tissue"]].to_string(index=False))
