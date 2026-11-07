#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Train one baseline on one CV fold of a manifest cohort.

    python scripts/train_baseline.py --model swinunetr --manifest mentor235 --fold 0
    python scripts/train_baseline.py --model eagan --manifest cohort_v2 --fold 2 --epochs 150

--manifest is a path or a name under $BASELINES_MANIFEST_DIR (default
$TAUGENNET_ROOT/data/manifests/<name>.csv). Checkpoints go to
results/checkpoints/baselines/<cohort>/<tag>/fold_<i>/{best,last}.pt (+ history.json); the
run refuses to overwrite an existing best.pt unless --overwrite is given.

Every hyperparameter has a per-model default (denseunet/registry.py) and can be overridden
on the command line; `--print-config` shows the resolved configuration and exits.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import CKPT_ROOT, MANIFEST_DIR, N_FOLDS, SEED  # noqa: E402
from denseunet.cohort import build_cohort  # noqa: E402
from denseunet import registry  # noqa: E402


def resolve_manifest(m):
    if os.path.exists(m):
        return os.path.abspath(m)
    cand = os.path.join(MANIFEST_DIR, m if m.endswith(".csv") else m + ".csv")
    if os.path.exists(cand):
        return cand
    sys.exit(f"Manifest not found: {m} (also tried {cand}). Run scripts/prepare_cohort.py first.")


def cohort_name(manifest_path):
    return os.path.splitext(os.path.basename(manifest_path))[0]


def run_dir(root, manifest_path, tag, fold):
    return os.path.join(root, cohort_name(manifest_path), tag, "single" if fold is None else f"fold_{fold}")


def main():
    p = argparse.ArgumentParser(description="Train an MRI→tau-PET baseline")
    p.add_argument("--model", choices=registry.model_names(), required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="only used without --fold and when the manifest has no fixed 'val' rows; "
                        "0.2 = CV fold 0, 0 = train on the whole pool")
    p.add_argument("--tag", default=None, help="run name (default = model name)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-cache", dest="cache", action="store_false", help="do not cache volumes in RAM")
    p.add_argument("--limit-subjects", type=int, default=None, help="debug: use only N train/test subjects")
    p.add_argument("--print-config", action="store_true")
    registry.add_model_args(p)
    args = registry.resolve_args(p.parse_args())
    args.tag = args.tag or args.model
    args.manifest = resolve_manifest(args.manifest)
    if args.print_config:
        print(json.dumps(vars(args), indent=1, default=str)); return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = run_dir(CKPT_ROOT, args.manifest, args.tag, args.fold)
    if os.path.exists(os.path.join(ckpt_dir, "best.pt")) and not args.overwrite:
        sys.exit(f"Refusing to overwrite {ckpt_dir}/best.pt (pass --overwrite).")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1, default=str)

    print(f"[train] model={args.model} tag={args.tag} cohort={cohort_name(args.manifest)} fold={args.fold} device={device}")
    print("[train] config: " + json.dumps({k: v for k, v in vars(args).items() if k not in ("manifest",)}, default=str))
    data = build_cohort(args.manifest, fold_idx=args.fold, n_folds=args.n_folds, val_frac=args.val_frac,
                        batch_size=args.batch_size, num_workers=args.num_workers, limit_subjects=args.limit_subjects)
    trainer = registry.build_trainer(args, device, cls_map=data["cls_map"])
    t0 = time.time()
    trainer.fit(data, ckpt_dir, cache=args.cache)
    print(f"[train] finished in {(time.time() - t0) / 60:.1f} min → {ckpt_dir}")


if __name__ == "__main__":
    main()
