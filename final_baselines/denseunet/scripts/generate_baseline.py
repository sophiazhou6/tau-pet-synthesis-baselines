#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Cache a trained baseline's predictions for the held-out test subjects in the exact
format TauGenNet's evaluator consumes: subject_{i:03d}.npy, normalized [0,1] PET space,
DK86-masked, i = position in the manifest's heldout_test rows (index.csv maps i → RID).

    python scripts/generate_baseline.py --model swinunetr --manifest mentor235 --fold 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import CKPT_ROOT, GEN_ROOT, N_FOLDS  # noqa: E402
from denseunet.cohort import build_cohort  # noqa: E402
from denseunet import registry  # noqa: E402
from denseunet import predictions as preds  # noqa: E402
from train_baseline import cohort_name, resolve_manifest, run_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=registry.model_names(), required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--tag", default=None)
    p.add_argument("--checkpoint", default=None, help="default: <ckpt run dir>/best.pt")
    p.add_argument("--out-dir", default=None, help="default: results/generated/baselines/<cohort>/<tag>/fold_<i>")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--limit-subjects", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.tag = args.tag or args.model
    args.manifest = resolve_manifest(args.manifest)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = args.checkpoint or os.path.join(run_dir(CKPT_ROOT, args.manifest, args.tag, args.fold), "best.pt")
    out_dir = args.out_dir or run_dir(GEN_ROOT, args.manifest, args.tag, args.fold)
    if args.split == "val":
        out_dir = out_dir + "_val"
    if preds.has_predictions(out_dir) and not args.overwrite:
        sys.exit(f"Refusing to overwrite cached generations in {out_dir} (pass --overwrite).")
    os.makedirs(out_dir, exist_ok=True)
    for stale in preds.npy_files(out_dir) + [os.path.join(out_dir, preds.PACK_NAME)]:   # only reached with --overwrite
        if os.path.exists(stale):
            os.remove(stale)

    state = torch.load(ckpt, map_location=device, weights_only=False)
    targs = argparse.Namespace(**state["args"])
    targs.model = args.model
    data = build_cohort(args.manifest, fold_idx=args.fold, n_folds=args.n_folds, batch_size=1,
                        num_workers=0, limit_subjects=args.limit_subjects)
    ds = data["test_ds"] if args.split == "test" else data["val_ds"]
    trainer = registry.build_trainer(targs, device, cls_map=state.get("cls_map"))
    trainer.load_state_dict(state["state"])
    trainer.attach_mask(ds)
    trainer.train_mode(False)
    print(f"[generate] {args.model} ckpt epoch {state.get('epoch')} val_mse {state.get('val_mse')}  "
          f"→ {len(ds)} {args.split} subjects → {out_dir}")

    with torch.no_grad():
        for i in range(len(ds)):
            pet, mri, _ = ds[i]
            pred = trainer.predict(mri.unsqueeze(0).to(device))
            np.save(os.path.join(out_dir, f"subject_{i:03d}.npy"), pred.squeeze().cpu().numpy().astype(np.float32))
    ds.index_table().to_csv(os.path.join(out_dir, "index.csv"), index=False)
    with open(os.path.join(out_dir, "generation.json"), "w") as f:
        json.dump({"model": args.model, "tag": args.tag, "checkpoint": ckpt, "epoch": state.get("epoch"),
                   "val_mse": state.get("val_mse"), "manifest": args.manifest, "fold": args.fold,
                   "split": args.split, "n": len(ds)}, f, indent=1)
    print(f"[generate] wrote {len(ds)} volumes")


if __name__ == "__main__":
    main()
