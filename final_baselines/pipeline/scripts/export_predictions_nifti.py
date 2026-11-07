#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Export a run's cached predictions as 3D NIfTI tau-PET volumes in SUVR units, one file per
subject, named by RID — the format Anil's downstream AD/MCI/CN classifier consumes.

    python scripts/export_predictions_nifti.py --cohort cohort_v2 --tag denseunet --fold 0 --split test
    python scripts/export_predictions_nifti.py --cohort cohort_v2 --all          # every tag/fold/split found

Details that matter:
  * predictions are stored normalized to [0,1] per subject. --scale fixed (default) converts them with
    one scale for every subject, the mean masked PET min/max of the split's training subjects, so no
    predicted subject's real scan is used. Written to <role>_fixed_scale/. --scale real uses each
    subject's OWN real PET masked min/max, as the evaluator does (dataset.get_pet_norms after access).
    Its files carry real-scan information (file max vs real max r = 0.92-0.99): never give them to a
    classifier;
  * the volumes live on the model grid (96×112×96); the NIfTI affine written here maps that grid
    exactly onto MNI space (derived from the subject's own PET header and the resize rule), so
    the files overlay correctly on MNI templates and the subject's real PET in any viewer;
  * reads loose subject_XXX.npy or the packed predictions.npz equally;
  * output: <out-root>/<tag>/fold_<f>/<test|val>/RID_<rid>_pred_tauPET_SUVR.nii.gz plus an
    index.csv (RID, group, role, fold, model) per folder and a top-level manifest.
"""
import argparse
import glob
import os
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from denseunet.config import GEN_ROOT, RESULTS_DIR, VOL_SHAPE  # noqa: E402
from denseunet.cohort import build_cohort  # noqa: E402
from denseunet import predictions as preds  # noqa: E402
from train_baseline import resolve_manifest  # noqa: E402


def resize_affine(ref_img, new_shape):
    """Affine of a volume produced by F.interpolate(align_corners=False) resizing ref_img to new_shape."""
    s = np.array(ref_img.shape[:3], float) / np.array(new_shape, float)
    M = np.eye(4)
    M[:3, :3] = np.diag(s)
    M[:3, 3] = 0.5 * s - 0.5
    return ref_img.affine @ M


def load_pred(gen_dir, i):
    f = os.path.join(gen_dir, f"subject_{i:03d}.npy")
    if os.path.exists(f):
        return np.load(f).astype(np.float32)
    z = getattr(load_pred, "_z", None)
    if z is None or getattr(load_pred, "_zdir", None) != gen_dir:
        load_pred._z = z = np.load(os.path.join(gen_dir, preds.PACK_NAME))
        load_pred._zdir = gen_dir
    return z[f"subject_{i:03d}"].astype(np.float32)


def pool_scale(data, fold):
    """Mean masked PET (min, max) over the split's training subjects: one scale for every prediction,
    taken from no scan being predicted. A refit trains on train + val, a fold model on train only."""
    lo, hi = [], []
    for ds in [data["train_ds"]] + ([data["val_ds"]] if fold is None else []):
        for i in range(len(ds)):
            ds[i]                                    # populates the masked per-subject norms
            a, b = ds.get_pet_norms(i)
            lo.append(float(a)); hi.append(float(b))
    return float(np.mean(lo)), float(np.mean(hi))


def export_one(manifest, tag, fold, role, out_root, cohort, scale="fixed", plasma_csv=None):
    sub = "single" if fold is None else f"fold_{fold}"
    gen_dir = os.path.join(GEN_ROOT, cohort, tag, sub + ("_val" if role == "val" else ""))
    if not preds.has_predictions(gen_dir):
        return None
    data = build_cohort(manifest, fold_idx=fold, batch_size=1, num_workers=0)
    ds = data["test_ds"] if role == "test" else data["val_ds"]
    idx_csv = os.path.join(gen_dir, "index.csv")
    if os.path.exists(idx_csv):
        import pandas as pd
        saved = [str(r) for r in pd.read_csv(idx_csv)["RID"]]
        if saved != [str(r) for r in ds.rids]:
            sys.exit(f"ABORT {gen_dir}: subject order differs from the generation run's index.csv "
                     f"({len(saved)} vs {len(ds)} subjects) — refusing to label volumes with the wrong RIDs")
    if preds.n_predictions(gen_dir) != len(ds):
        sys.exit(f"ABORT {gen_dir}: {preds.n_predictions(gen_dir)} saved volumes but {len(ds)} subjects in the split")
    out_dir = os.path.join(out_root, tag, sub or "single", role + ("_fixed_scale" if scale == "fixed" else ""))
    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd
    extra = dict(scale=scale)
    if scale == "fixed":
        pmin, pmax = pool_scale(data, fold)
        extra.update(scale_min=round(pmin, 6), scale_max=round(pmax, 6))
        print(f"[{tag} fold {fold} {role}] fixed scale {pmin:.6f}–{pmax:.6f} SUVR (training subjects)", flush=True)
    imputed = {}
    if plasma_csv:                                   # which subjects' plasma was MICE-imputed (TauGenNet)
        imputed = {str(int(r.RID)): int(r.imputed) for r in pd.read_csv(plasma_csv).itertuples()}
    rows = []
    for i in range(len(ds)):
        pet_ref = nib.load(ds.pet_paths[i])          # header only when the scale is fixed
        if scale == "real":
            ds[i]                                    # populates the masked per-subject norms
            pmin, pmax = ds.get_pet_norms(i)
        suvr = (load_pred(gen_dir, i) * (pmax - pmin + 1e-8) + pmin).astype(np.float32)
        if scale == "fixed":                         # the scorer's DK86 mask: one atlas, same coverage for every model
            suvr = suvr * (ds._dk_mask.squeeze().numpy() > 0)
        aff = resize_affine(pet_ref, VOL_SHAPE)
        name = f"RID_{ds.rids[i]}_pred_tauPET_SUVR.nii.gz"
        nib.save(nib.Nifti1Image(suvr, aff), os.path.join(out_dir, name))
        rows.append(dict(RID=ds.rids[i], group=ds.groups[i], role=role, fold=fold, model=tag, file=name, **extra))
        if plasma_csv:
            rows[-1]["plasma_imputed"] = imputed[str(int(ds.rids[i]))]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "index.csv"), index=False)
    print(f"[{tag} fold {fold} {role}] {len(rows)} NIfTIs → {out_dir}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="cohort_v2")
    p.add_argument("--manifest", default=None)
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--fold", type=int, action="append", default=[])
    p.add_argument("--split", choices=["test", "val", "both"], default="both")
    p.add_argument("--all", action="store_true", help="export every tag/fold with cached predictions")
    p.add_argument("--scale", choices=["fixed", "real"], default="fixed",
                   help="fixed: one training-pool scale for every subject; real: each subject's own real PET range")
    p.add_argument("--plasma-csv", default=None,
                   help="per-split MICE table (scripts/impute_plasma.py): adds a plasma_imputed column to index.csv")
    p.add_argument("--out-root", default=None,
                   help="default: results/export_nifti/<cohort>/ (share this tree with the lab)")
    a = p.parse_args()
    manifest = resolve_manifest(a.manifest or a.cohort)
    out_root = a.out_root or os.path.join(RESULTS_DIR, "export_nifti", a.cohort)
    base = os.path.join(GEN_ROOT, a.cohort)
    if a.all:
        combos = []
        for d in sorted(glob.glob(os.path.join(base, "*", "fold_*")) + glob.glob(os.path.join(base, "*", "single*"))):
            tag, sub = d.split(os.sep)[-2:]
            fold = int(sub.split("_")[1]) if sub.startswith("fold_") and not sub.endswith("_val") else \
                   (int(sub.split("_")[1]) if sub.startswith("fold_") else None)
            role = "val" if sub.endswith("_val") else "test"
            if preds.has_predictions(d):
                combos.append((tag, fold, role))
    else:
        roles = ["test", "val"] if a.split == "both" else [a.split]
        combos = [(t, f, r) for t in (a.tag or sys.exit("--tag or --all")) for f in (a.fold or [None]) for r in roles]
    allrows = []
    for tag, fold, role in combos:
        rows = export_one(manifest, tag, fold, role, out_root, a.cohort, a.scale, a.plasma_csv)
        if rows:
            allrows += rows
    if allrows:
        import pandas as pd
        parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(out_root, "*", "*", "*", "index.csv")))]
        allx = pd.concat(parts, ignore_index=True)
        allx.to_csv(os.path.join(out_root, "manifest_all.csv"), index=False)
        print(f"wrote {len(allrows)} volumes this run; export tree now holds {len(allx)} → {out_root}")


if __name__ == "__main__":
    main()
