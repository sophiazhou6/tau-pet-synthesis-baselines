# failure signature per subject: mean intensity in the outer 3-voxel shell of the volume
# (a real or healthy prediction is ~0 there; fold 1 had a bright rim)
import numpy as np
R = "/scratch/network/sz3962/tau-denseunet-baseline/results/generated/baselines/cohort_v2/taugennet_paper"
for d in [f"fold_{k}{s}" for k in range(5) for s in ("", "_val")] + ["single"]:
    z = np.load(f"{R}/{d}/predictions.npz")
    rims = []; mx = []
    for k in z.files:
        if not k.startswith("subject_"): continue
        g = z[k].squeeze()
        rim = np.ones_like(g, bool); rim[3:-3, 3:-3, 3:-3] = False
        rims.append(g[rim].mean()); mx.append(np.abs(g[rim]).max())
    rims = np.array(rims)
    print(f"{d:11s} n={len(rims):3d}  rim mean: median {np.median(rims):.4f}  max {rims.max():.4f}  subjects>0.05: {(rims>0.05).sum():3d}  rim abs max {max(mx):.3f}")
