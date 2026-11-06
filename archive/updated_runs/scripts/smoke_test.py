#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Build every registered model, run one real training step of its trainer on synthetic
volumes at VOL_SHAPE, then predict — reports time and peak GPU memory. Catches shape /
device / loss bugs before any SLURM time is spent.

    python scripts/smoke_test.py                       # all models, CPU or GPU if visible
    python scripts/smoke_test.py --models eagan ct2pet --batch-size 2
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines import config as _cfg  # noqa: E402
from baselines import registry  # noqa: E402
VOL_SHAPE = tuple(_cfg.VOL_SHAPE)


class FakeVolumes(Dataset):
    def __init__(self, n, mask, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for i in range(n):
            pet = torch.rand((1,) + tuple(VOL_SHAPE), generator=g) * mask
            mri = torch.rand((1,) + tuple(VOL_SHAPE), generator=g) * mask
            self.items.append((pet, mri, torch.tensor([float(i % 2)])))
        self.labels = [i % 2 for i in range(n)]
        self._dk_mask = mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def ball_mask():
    H, W, D = VOL_SHAPE
    z, y, x = np.mgrid[:H, :W, :D]
    r = ((z - H / 2) / (H * 0.38)) ** 2 + ((y - W / 2) / (W * 0.4)) ** 2 + ((x - D / 2) / (D * 0.38)) ** 2
    return torch.from_numpy((r < 1).astype(np.float32)).unsqueeze(0)


def run_one(model, device, batch_size, n_items, overrides):
    args = argparse.Namespace(model=model, seed=0, batch_size=batch_size, epochs=2, patience=0, num_workers=0)
    for k, v in registry.defaults_for(model).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.batch_size = batch_size
    args.n_slices = VOL_SHAPE[2]
    for k, v in overrides.items():
        cur = getattr(args, k, None)
        setattr(args, k, type(cur)(v) if cur is not None and not isinstance(cur, bool) else (v in ("1", "true", "True")))
    for k in ("epochs_share", "epochs_cls", "epochs_joint"):
        setattr(args, k, 1)
    mask = ball_mask()
    ds = FakeVolumes(n_items, mask)
    tr = registry.build_trainer(args, device, cls_map={"MCI": 0, "AD": 1})
    tr.attach_mask(ds)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stages = ["A", "B", "C"] if model.startswith("jointsyn") else [None]
    for st in stages:
        if st:
            tr._set_stage_optimizers(st)
        loader = tr.make_train_loader(ds)
        losses = tr.train_epoch(loader, 1)
        print(f"    {model}{'/' + st if st else ''} train_epoch: " + "  ".join(f"{k} {v:.4g}" for k, v in losses.items()))
    t_train = time.time() - t0
    tr.train_mode(False)
    t1 = time.time()
    with torch.no_grad():
        pred = tr.predict(ds[0][1].unsqueeze(0).to(device))
    t_pred = time.time() - t1
    assert tuple(pred.shape) == (1, 1) + tuple(VOL_SHAPE), pred.shape
    assert torch.isfinite(pred).all(), "non-finite prediction"
    bg = pred[0, 0][mask[0] == 0]
    assert bg.abs().max().item() < 1e-6, "prediction not masked to background"
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.startswith("cuda") else float("nan")
    val = tr.validate(ds)
    print(f"  OK {model:16s} train {t_train:6.1f}s ({n_items} vols, bs {batch_size})  predict {t_pred:5.2f}s  "
          f"peak mem {mem:5.1f} GB  val_mse {val:.4f}  pred range [{pred.min():.3f}, {pred.max():.3f}]")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=registry.model_names())
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--n-items", type=int, default=4)
    p.add_argument("--vol-shape", default=None, help="e.g. 32,64,32 for a quick CPU check (default: lab VOL_SHAPE)")
    p.add_argument("--override", nargs="*", default=[], help="k=v hyperparameter overrides, e.g. num_downs=5")
    a = p.parse_args()
    global VOL_SHAPE
    if a.vol_shape:
        VOL_SHAPE = tuple(int(v) for v in a.vol_shape.split(","))
    overrides = dict(kv.split("=", 1) for kv in a.override)
    print(f"smoke test on {a.device}, VOL_SHAPE {VOL_SHAPE}, torch {torch.__version__}")
    failed = []
    for m in a.models:
        print(f"\n=== {m}: {registry.MODELS[m]['description']}")
        try:
            run_one(m, a.device, a.batch_size, a.n_items, overrides)
        except Exception:
            failed.append(m); traceback.print_exc()
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print("\nFAILED: " + (", ".join(failed) if failed else "none"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
