"""
Lossless packing of cached predictions. A prediction volume is DK86-masked (≈91 % zeros), so
`subject_XXX.npy` files compress ≈11×: one run's test predictions go from ~4 MiB per subject to
a single `predictions.npz`. Scoring needs the plain .npy layout (it is what the GitHub
evaluate_final.py reads), so score_baseline.py unpacks to a temporary directory when only the
archive is present and packs again after a successful score.
"""
import glob
import json
import os
import re
import shutil
import tempfile
import time

import numpy as np

PACK_NAME = "predictions.npz"
_PAT = re.compile(r"subject_\d{3}\.npy$")


def npy_files(gen_dir):
    return sorted(f for f in glob.glob(os.path.join(gen_dir, "subject_*.npy")) if _PAT.search(f))


def is_packed(gen_dir):
    return os.path.exists(os.path.join(gen_dir, PACK_NAME))


def has_predictions(gen_dir):
    return os.path.isdir(gen_dir) and (bool(npy_files(gen_dir)) or is_packed(gen_dir))


def n_predictions(gen_dir):
    """Number of distinct subject volumes saved in gen_dir, loose or packed."""
    names = {os.path.basename(f)[:-4] for f in npy_files(gen_dir)}
    if is_packed(gen_dir):
        with np.load(os.path.join(gen_dir, PACK_NAME)) as z:
            names |= set(z.files)
    return len(names)


def is_complete(gen_dir, quiet_s=120):
    """True when every volume the generator announced (generation.json "n") is on disk and nothing
    was written for quiet_s seconds — i.e. no job is still filling the directory."""
    gj = os.path.join(gen_dir, "generation.json")
    try:
        n = json.load(open(gj))["n"]
    except (OSError, ValueError, KeyError):
        return False
    if n_predictions(gen_dir) != n:
        return False
    newest = max(os.path.getmtime(f) for f in npy_files(gen_dir) + [gj])
    return time.time() - newest > quiet_s


def pack(gen_dir, verbose=True):
    """subject_*.npy → predictions.npz (verified bit-identical), then remove the .npy files.
    Returns (bytes_before, bytes_after) or None when there is nothing to pack."""
    files = npy_files(gen_dir)
    if not files:
        return None
    before = sum(os.path.getsize(f) for f in files)
    arrs = {os.path.basename(f)[:-4]: np.load(f) for f in files}
    final, tmp = os.path.join(gen_dir, PACK_NAME), os.path.join(gen_dir, PACK_NAME + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrs)
    with np.load(tmp) as z:
        if sorted(z.files) != sorted(arrs) or not all(
                z[k].dtype == arrs[k].dtype and np.array_equal(z[k], arrs[k]) for k in arrs):
            os.remove(tmp)
            raise IOError(f"packed archive does not reproduce the predictions in {gen_dir}; nothing removed")
    os.replace(tmp, final)
    for f in files:
        os.remove(f)
    after = os.path.getsize(final)
    if verbose:
        print(f"[pack] {gen_dir}: {len(files)} volumes, {before / 2**20:.0f} MiB → {after / 2**20:.0f} MiB")
    return before, after


def unpack(gen_dir, dest=None):
    """predictions.npz → subject_*.npy in `dest` (default: a new temp dir). Returns dest."""
    src = os.path.join(gen_dir, PACK_NAME)
    if dest is None:
        dest = tempfile.mkdtemp(prefix="tau_pred_", dir=os.environ.get("TMPDIR") or "/tmp")
    os.makedirs(dest, exist_ok=True)
    with np.load(src) as z:
        for k in z.files:
            np.save(os.path.join(dest, k + ".npy"), z[k])
    for extra in ("index.csv", "generation.json"):
        if os.path.exists(os.path.join(gen_dir, extra)) and os.path.abspath(dest) != os.path.abspath(gen_dir):
            shutil.copy(os.path.join(gen_dir, extra), os.path.join(dest, extra))
    return dest
