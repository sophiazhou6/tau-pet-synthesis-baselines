#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Reclaim scratch space from finished runs: delete resume-only / duplicate checkpoint files,
keeping exactly what generation and analysis need (best.pt per run; for the joint framework
also stageA/best.pt = the synthesis-only model and stageB/best.pt = the pretrained classifier),
and pack cached predictions losslessly (subject_*.npy → predictions.npz).

    python scripts/prune_checkpoints.py --cohort mentor235            # dry run: lists what would go
    python scripts/prune_checkpoints.py --cohort mentor235 --apply
    python scripts/prune_checkpoints.py --cohort mentor235 --apply --drop-stages   # also stageA/stageB

Only runs whose scoring has finished (metrics.json present) are touched, so a training job in
progress is never pruned; a prediction directory is packed only once it is complete (every
announced volume on disk, nothing written for 2 minutes), so a job still writing it is left alone.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.config import CKPT_ROOT, GEN_ROOT, RECORDS_ROOT  # noqa: E402
from baselines import predictions as preds  # noqa: E402


def prune_run(cohort, tag, fold_name, apply=False, drop_stages=False, verbose=True):
    """Prune one scored run (fold_name like 'fold_0'); returns (bytes deleted or deletable, dirs packed)."""
    if not os.path.exists(os.path.join(RECORDS_ROOT, cohort, tag, fold_name, "metrics.json")):
        return 0, []
    run = os.path.join(CKPT_ROOT, cohort, tag, fold_name)
    victims = [os.path.join(run, f) for f in ("last.pt", "stageA/last.pt", "stageC/last.pt")
               if os.path.exists(os.path.join(run, f))]
    c_best, best = os.path.join(run, "stageC", "best.pt"), os.path.join(run, "best.pt")
    if os.path.exists(c_best) and os.path.exists(best):
        victims.append(c_best)
    if drop_stages:
        victims += [os.path.join(run, f) for f in ("stageA/best.pt", "stageB/best.pt") if os.path.exists(os.path.join(run, f))]
    total = sum(os.path.getsize(v) for v in victims)
    for v in victims:
        if verbose:
            print(f"{os.path.getsize(v) / 2**20:8.0f} MB  {v}")
        if apply:
            os.remove(v)
    packed = []
    for gen in (os.path.join(GEN_ROOT, cohort, tag, fold_name), os.path.join(GEN_ROOT, cohort, tag, fold_name + "_val")):
        if preds.npy_files(gen) and preds.is_complete(gen):
            if verbose:
                n = sum(os.path.getsize(f) for f in preds.npy_files(gen))
                print(f"{n / 2**20:8.0f} MB  {gen}/subject_*.npy  → pack (lossless, ~11x smaller)")
            if apply:
                preds.pack(gen, verbose=False)
            packed.append(gen)
    return total, packed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--drop-stages", action="store_true", help="also delete the joint framework's stage A/B checkpoints")
    a = p.parse_args()
    runs = sorted({tuple(d.split(os.sep)[-2:]) for d in glob.glob(os.path.join(RECORDS_ROOT, a.cohort, "*", "fold_*"))
                   if not d.endswith("_val")})
    total = 0
    for tag, fold_name in runs:
        total += prune_run(a.cohort, tag, fold_name, apply=a.apply, drop_stages=a.drop_stages)[0]
    print(f"\ncheckpoints: {total / 2**30:.2f} GB {'deleted' if a.apply else 'would be deleted (add --apply)'}"
          f"{'' if a.apply else '; prediction dirs listed above would be packed'}")


if __name__ == "__main__":
    main()
