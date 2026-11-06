"""
BaseTrainer — the contract every baseline follows so that train/generate/score scripts
are model-agnostic:

  build()                      create networks + optimizers (self.nets: dict name → nn.Module)
  make_train_loader(cached)    DataLoader over the cached training volumes (default: paired)
  train_epoch(loader, epoch)   one pass; returns {loss_name: mean}
  predict(mri)                 (N,1,H,W,D) MRI in [0,1] → tau PET in [0,1], DK86-masked
  on_epoch_end(epoch)          LR schedules
  state_dict()/load_state_dict()

fit() owns: RAM caching of volumes, the validation metric (masked MSE of predict() vs the
paired PET in [0,1] space — identical for every model, so checkpoints are always selected
by the same criterion, on the validation split only), best/last checkpointing, early
stopping, history.json. Test subjects are never touched during training.
"""
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..cohort import CachedVolumes
from ..config import VOL_SHAPE
from ..models.common import count_params


def masked_mse(pred, target, mask):
    se = (pred - target) ** 2 * mask
    return se.sum() / (mask.sum() * pred.shape[0]).clamp(min=1)


def masked_l1(pred, target, mask):
    ae = (pred - target).abs() * mask
    return ae.sum() / (mask.sum() * pred.shape[0]).clamp(min=1)


class BaseTrainer:
    name = "base"

    def __init__(self, args, device, cls_map=None):
        self.args = args
        self.device = torch.device(device)
        self.cls_map = cls_map or {}
        self.mask = None            # (1,1,H,W,D) float, set by fit()/attach_mask()
        self.nets = {}
        self.optimizers = {}
        self.epoch = 0
        self.build()
        for k, n in self.nets.items():
            print(f"[{self.name}] {k}: {count_params(n) / 1e6:.2f}M params", flush=True)

    # ── to implement ───────────────────────────────────────────────────────────
    def build(self):
        raise NotImplementedError

    def train_epoch(self, loader, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, mri):
        raise NotImplementedError

    def on_epoch_end(self, epoch):
        pass

    def make_train_loader(self, cached):
        return DataLoader(cached, batch_size=self.args.batch_size, shuffle=True, drop_last=False,
                          num_workers=0, pin_memory=True)

    def extra_val_metrics(self, val_cached):
        return {}

    # ── shared machinery ──────────────────────────────────────────────────────
    def attach_mask(self, ds):
        m = ds._dk_mask
        if m is None:
            m = torch.ones((1,) + tuple(VOL_SHAPE))
        self.mask = m.float().unsqueeze(0).to(self.device)   # (1,1,H,W,D)

    def state_dict(self):
        return {k: n.state_dict() for k, n in self.nets.items()}

    def load_state_dict(self, sd, strict=True):
        for k, n in self.nets.items():
            if k in sd:
                n.load_state_dict(sd[k], strict=strict)
            else:
                print(f"[{self.name}] warning: checkpoint has no weights for '{k}'")

    def train_mode(self, flag=True):
        for n in self.nets.values():
            n.train(flag)

    @torch.no_grad()
    def validate(self, val_cached, batch_size=None):
        """Masked MSE (in [0,1] PET space) of predict() over the validation subjects."""
        if val_cached is None or len(val_cached) == 0:
            return float("nan")
        self.train_mode(False)
        bs = batch_size or max(1, self.args.batch_size)
        loader = DataLoader(val_cached, batch_size=bs, shuffle=False, num_workers=0)
        tot, n = 0.0, 0
        for pet, mri, _ in loader:
            pet, mri = pet.to(self.device), mri.to(self.device)
            pred = self.predict(mri)
            tot += masked_mse(pred, pet, self.mask).item() * pet.shape[0]
            n += pet.shape[0]
        self.train_mode(True)
        return tot / max(n, 1)

    def cached_data(self, data, cache=True):
        """(train, val) volume sets, materialised in RAM once per trainer and reused across stages."""
        key = id(data)
        store = getattr(self, "_cache_store", None)
        if store is not None and store[0] == key:
            return store[1], store[2]
        t0 = time.time()
        print(f"[{self.name}] caching {len(data['train_ds'])} train + {len(data['val_ds'])} val volumes…", flush=True)
        train_c = CachedVolumes(data["train_ds"]) if cache else data["train_ds"]
        val_c = CachedVolumes(data["val_ds"], verbose=False) if len(data["val_ds"]) else None
        print(f"[{self.name}] cached in {time.time() - t0:.0f}s", flush=True)
        self._cache_store = (key, train_c, val_c)
        return train_c, val_c

    def save(self, path, epoch, val, history, extra=None):
        payload = {"trainer": self.name, "args": vars(self.args), "epoch": epoch, "val_mse": val,
                   "cls_map": self.cls_map, "state": self.state_dict(), "history": history}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def fit(self, data, ckpt_dir, epochs=None, patience=None, cache=True, log_every=1):
        args = self.args
        epochs = epochs or args.epochs
        patience = args.patience if patience is None else patience
        os.makedirs(ckpt_dir, exist_ok=True)
        self.attach_mask(data["train_ds"])

        train_c, val_c = self.cached_data(data, cache)
        loader = self.make_train_loader(train_c)

        best_path, last_path = os.path.join(ckpt_dir, "best.pt"), os.path.join(ckpt_dir, "last.pt")
        history, best_val, bad = [], float("inf"), 0
        self.train_mode(True)
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            te = time.time()
            losses = self.train_epoch(loader, epoch)
            self.on_epoch_end(epoch)
            val = self.validate(val_c)
            extra = self.extra_val_metrics(val_c) if val_c is not None else {}
            rec = {"epoch": epoch, "val_mse": val, "time_s": time.time() - te, **losses, **extra}
            history.append(rec)
            flag = ""
            save_last = bool(getattr(args, "save_last", False))
            if val_c is None:
                self.save(best_path, epoch, val, history)      # no validation → the last epoch is the model
            else:
                if save_last:                                  # resume aid only; doubles the disk footprint
                    self.save(last_path, epoch, val, history)
                if val < best_val:
                    best_val, bad, flag = val, 0, "  *best"
                    self.save(best_path, epoch, val, history)
                else:
                    bad += 1
            if epoch % log_every == 0:
                ls = "  ".join(f"{k} {v:.4g}" for k, v in losses.items())
                ex = "  ".join(f"{k} {v:.4g}" for k, v in extra.items())
                print(f"[{self.name}] epoch {epoch:3d}/{epochs}  {ls}  val_mse {val:.5f}  {ex}"
                      f"  ({rec['time_s']:.0f}s){flag}", flush=True)
            with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=1)
            if val_c is not None and patience and bad >= patience:
                print(f"[{self.name}] early stopping at epoch {epoch} (no val improvement for {patience}).")
                break
        print(f"[{self.name}] done. best val_mse {best_val:.5f} → {best_path}", flush=True)
        return history


def linear_decay_lambda(n_const, n_decay):
    """pix2pix 'linear' policy: lr constant for n_const epochs, then linearly → 0 over n_decay."""
    def f(epoch):  # epoch is 0-based scheduler step
        return 1.0 - max(0, epoch - n_const) / float(max(n_decay, 1))
    return f


def warmup_cosine_lambda(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / float(max(warmup, 1))
        p = (step - warmup) / float(max(total - warmup, 1))
        return 0.5 * (1.0 + np.cos(np.pi * min(p, 1.0)))
    return f
