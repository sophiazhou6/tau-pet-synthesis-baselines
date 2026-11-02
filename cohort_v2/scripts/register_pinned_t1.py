#!/usr/bin/env python3
"""
Register one pinned raw T1 to MNI152 2 mm, reproducing the lab's own pipeline: FLIRT affine
to MNI152_T1_2mm, then FNIRT with exactly the parameters recorded in the lab's
*_to_MNI152_T1_2mm.log files next to each raw scan (= FSL's stock T1_2_MNI152_2mm settings).
Output: <out-root>/<PTID>/T1_to_MNI_nonlin.nii.gz  (the file name every model expects),
plus affine.mat and warpcoef.nii.gz so the matching segmentation can be warped later.

    python scripts/register_pinned_t1.py --index 0..596          # one row of pinned_scans_597.csv
    python scripts/register_pinned_t1.py --rid 6644 --rid 4229   # specific subjects
Resumable: a subject whose output exists and loads is skipped. Stdlib only (runs in the fslreg env).
"""
import argparse, csv, os, shutil, subprocess, sys, time

TG = os.environ.get("TAUGENNET_ROOT", "/scratch/network/sz3962/taugennet")
BOX = f"{TG}/data/box"
PIN = "/scratch/network/sz3962/tau-denseunet-baseline/results/for_anil/pinned_scans_597.csv"
FSL = os.environ.get("FSLREG_ENV", "/scratch/network/sz3962/.conda/envs/fslreg")
STD = f"{FSL}/data/standard"
FNIRT_ARGS = [  # verbatim from the lab's 2mm logs (== stock T1_2_MNI152_2mm config)
    f"--refmask={STD}/MNI152_T1_2mm_brain_mask_dil", "--imprefm=1", "--impinm=1", "--imprefval=0",
    "--impinval=0", "--subsamp=4,4,2,2,1,1", "--miter=5,5,5,5,5,10", "--infwhm=8,6,5,4.5,3,2",
    "--reffwhm=8,6,5,4,2,0", "--lambda=300,150,100,50,40,30", "--estint=1,1,1,1,1,0",
    "--applyrefmask=1,1,1,1,1,1", "--applyinmask=1",   # in the stock T1_2_MNI152_2mm.cnf; fnirt omits them from its logs
    "--warpres=10,10,10", "--splineorder=3", "--ssqlambda=1", "--jacrange=0.01,100",
    "--regmod=bending_energy", "--intmod=global_non_linear_with_bias", "--intorder=5",
    "--biasres=50,50,50", "--biaslambda=10000", "--numprec=double", "--interp=linear", "--refderiv=0"]


def run(cmd):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "FSLDIR": FSL, "FSLOUTPUTTYPE": "NIFTI_GZ",
                                                                 "PATH": f"{FSL}/bin:" + os.environ.get("PATH", "")})
    print(f"  [{time.time() - t:6.0f}s] {' '.join(cmd[:2])} → exit {r.returncode}", flush=True)
    if r.returncode:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
    return r.returncode


def canonicalise(raw, out_dir):
    """Rewrite the scan with standard (RAS) axis order/direction — fixes scans whose header
    direction labels are wrong, which send FLIRT to a mirrored/rotated optimum. Uses the
    taugennet python (nibabel). NOTE: warpcoef then maps the CANONICAL image to MNI; warping
    the matching seg later must apply the same reorientation first."""
    canon = os.path.join(out_dir, "_canon.nii.gz")
    code = ("import nibabel as nib, sys; im = nib.load(sys.argv[1]); "
            "nib.save(nib.as_closest_canonical(im), sys.argv[2])")
    if subprocess.run(["/home/sz3962/.conda/envs/taugennet/bin/python3", "-c", code, raw, canon]).returncode:
        return None
    return canon


def robust_affine(raw, out_dir, full_search=False):
    """Neck-crop + skull-strip before FLIRT (the fsl_anat recipe) — for scans where plain FLIRT
    mis-initialises. Returns the raw→MNI affine path."""
    crop, r2f = os.path.join(out_dir, "_crop.nii.gz"), os.path.join(out_dir, "_roi2full.mat")
    brain, f2r = os.path.join(out_dir, "_brain.nii.gz"), os.path.join(out_dir, "_full2roi.mat")
    b2m, aff = os.path.join(out_dir, "_brain2mni.mat"), os.path.join(out_dir, "affine.mat")
    if run(["robustfov", "-i", raw, "-r", crop, "-m", r2f]):
        return None
    if run(["bet2", crop, brain.replace(".nii.gz", ""), "-f", "0.3"]):
        return None
    search = ["-searchrx", "-180", "180", "-searchry", "-180", "180", "-searchrz", "-180", "180"] if full_search else []
    if run(["flirt", "-in", brain, "-ref", f"{STD}/MNI152_T1_2mm_brain.nii.gz", "-omat", b2m, "-dof", "12"] + search):
        return None
    if run(["convert_xfm", "-omat", f2r, "-inverse", r2f]) or        run(["convert_xfm", "-omat", aff, "-concat", b2m, f2r]):
        return None
    for f in (crop, brain, r2f, f2r, b2m):
        if os.path.exists(f):
            os.remove(f)
    return aff


def one(row, out_root, robust=False, overwrite=False, rescue=False):
    ptid, rid = row["PTID"], row["RID"]
    raw = os.path.join(BOX, row["box_dir"], row["mri_file"])
    out_dir = os.path.join(out_root, ptid)
    final = os.path.join(out_dir, "T1_to_MNI_nonlin.nii.gz")
    if os.path.exists(final) and os.path.getsize(final) > 1e6 and not overwrite:
        print(f"[{ptid} / RID {rid}] exists, skipping"); return 0
    if not os.path.exists(raw):
        sys.stderr.write(f"[{ptid}] raw scan missing: {raw}\n"); return 2
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{ptid} / RID {rid}] {row['mri_file']} ({row.get('pinned_date', '?')}) → {final}", flush=True)
    aff, warp, tmp = os.path.join(out_dir, "affine.mat"), os.path.join(out_dir, "warpcoef.nii.gz"), final + ".tmp.nii.gz"
    if rescue:                                   # header-direction repair + exhaustive rotation search
        canon = canonicalise(raw, out_dir)
        if canon is None or robust_affine(canon, out_dir, full_search=True) is None:
            return 3
        raw = canon                              # fnirt must warp the reoriented image
    elif robust:
        if robust_affine(raw, out_dir) is None:
            return 3
    elif run(["flirt", "-in", raw, "-ref", f"{STD}/MNI152_T1_2mm.nii.gz", "-omat", aff, "-dof", "12"]):
        return 3
    if run(["fnirt", f"--ref={STD}/MNI152_T1_2mm.nii.gz", f"--in={raw}", f"--aff={aff}",
            f"--cout={warp}", f"--iout={tmp}"] + FNIRT_ARGS):
        return 4
    os.replace(tmp, final)
    canon = os.path.join(out_dir, "_canon.nii.gz")
    if os.path.exists(canon) and not rescue:
        os.remove(canon)
    if rescue and os.path.exists(canon):
        os.remove(canon)                         # drop the temp; warpcoef provenance noted in canonicalise()
    print(f"[{ptid}] done", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--rid", action="append", default=[])
    p.add_argument("--out-root", default=f"{TG}/data/mni_t1")
    p.add_argument("--robust", action="store_true", help="neck-crop + skull-strip before the affine (retry mode)")
    p.add_argument("--pin-csv", default=None, help="alternative subject table (same columns as pinned_scans_597.csv)")
    p.add_argument("--rescue", action="store_true", help="reorient header to canonical + full rotation search (last resort)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.pin_csv or PIN)))
    todo = [rows[a.index]] if a.index is not None else [r for r in rows if r["RID"] in set(a.rid)]
    if not todo:
        sys.exit("nothing selected (--index or --rid)")
    rc = 0
    for r in todo:
        rc = max(rc, one(r, a.out_root, robust=a.robust, overwrite=a.overwrite, rescue=a.rescue))
    sys.exit(rc)


if __name__ == "__main__":
    main()
