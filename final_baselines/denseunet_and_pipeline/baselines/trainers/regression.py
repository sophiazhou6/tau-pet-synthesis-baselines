"""
Deterministic regressors trained with the DK86-masked MSE of the Dense-U-Net baseline.

  denseunet  Adam, lr 1e-4, no weight decay (the summer configuration; grid-search winner
             lr 1e-4 / wd 0), no augmentation.
  swinunetr  AdamW, lr 1e-4, weight decay 1e-5, linear warm-up + cosine decay, AMP —
             the official Swin UNETR BraTS recipe (research-contributions/SwinUNETR:
             optim_lr 1e-4, reg_weight 1e-5, warmup_cosine; the paper text quotes 8e-4
             for its 8-GPU DGX run). Augmentation as in the paper but restricted to what
             is anatomically valid for MNI-registered volumes: random left-right flip
             (p=0.5; the paper flips all three axes) and random intensity shift (±0.1) /
             scale (0.9–1.1) of the input MRI. Both are switchable (--no-augment).
"""
import copy
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseTrainer, masked_mse, warmup_cosine_lambda
from ..models.denseunet import DenseUNet3D
from ..models.swin_unetr import SwinUNETRRegressor


class RegressionTrainer(BaseTrainer):
    name = "regression"

    def build(self):
        a = self.args
        if a.model in ("denseunet", "denseunet_paper"):
            net = DenseUNet3D(in_ch=1, out_ch=1)
            if a.init == "glorot":                      # Keras defaults of the released model
                for m in net.modules():
                    if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
                        torch.nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias)
        elif a.model == "swinunetr":
            net = SwinUNETRRegressor(in_ch=1, out_ch=1, feature_size=a.swin_feature_size,
                                     drop_path_rate=a.swin_drop_path, use_checkpoint=a.swin_checkpoint)
        else:
            raise ValueError(a.model)
        self.nets["G"] = net.to(self.device)
        if a.optimizer == "adamw":
            opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        elif a.weight_decay > 0:
            opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        else:
            opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
        self.optimizers["G"] = opt
        self.sched = None
        self.use_amp = bool(a.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._steps_per_epoch = None

    def _augment(self, mri, pet):
        a = self.args
        if not a.augment:
            return mri, pet
        if a.flip_lr and torch.rand(1).item() < 0.5:
            mri, pet = mri.flip(2), pet.flip(2)          # spatial dim 0 = MNI x (left-right)
        if a.intensity_jitter:
            n = mri.shape[0]
            shift = (torch.rand(n, 1, 1, 1, 1, device=mri.device) * 0.2 - 0.1)
            scale = (torch.rand(n, 1, 1, 1, 1, device=mri.device) * 0.2 + 0.9)
            mri = (mri * scale + shift) * self.mask     # keep background exactly 0
        return mri, pet

    def train_epoch(self, loader, epoch):
        a = self.args
        if self.sched is None and a.schedule == "warmup_cosine":
            total = a.epochs * len(loader)
            self.sched = torch.optim.lr_scheduler.LambdaLR(
                self.optimizers["G"], warmup_cosine_lambda(total, int(a.warmup_frac * total)))
        net, opt = self.nets["G"], self.optimizers["G"]
        tot, n = 0.0, 0
        for pet, mri, _ in loader:
            pet, mri = pet.to(self.device, non_blocking=True), mri.to(self.device, non_blocking=True)
            mri, pet = self._augment(mri, pet)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                pred = net(mri)
            loss = (F.mse_loss(pred.float(), pet) if a.loss == "mse_full"
                    else masked_mse(pred.float(), pet, self.mask))
            self.scaler.scale(loss).backward()
            if a.grad_clip > 0:
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), a.grad_clip)
            self.scaler.step(opt)
            self.scaler.update()
            if self.sched is not None:
                self.sched.step()
            tot += loss.item() * pet.shape[0]
            n += pet.shape[0]
        return {"train_mse": tot / max(n, 1), "lr": opt.param_groups[0]["lr"]}

    @torch.no_grad()
    def validate(self, val_cached, batch_size=None):
        if self.args.loss != "mse_full":
            return super().validate(val_cached, batch_size)
        # Keras val_loss of the released model: whole-volume MSE of the raw network output
        if val_cached is None or len(val_cached) == 0:
            return float("nan")
        self.train_mode(False)
        loader = DataLoader(val_cached, batch_size=batch_size or self.args.batch_size, shuffle=False, num_workers=0)
        tot, n = 0.0, 0
        for pet, mri, _ in loader:
            pet, mri = pet.to(self.device), mri.to(self.device)
            tot += F.mse_loss(self.nets["G"](mri).float(), pet).item() * pet.shape[0]
            n += pet.shape[0]
        self.train_mode(True)
        return tot / max(n, 1)

    def fit(self, data, ckpt_dir, epochs=None, patience=None, cache=True, log_every=1):
        a = self.args
        if a.schedule != "upstream_rounds":
            return super().fit(data, ckpt_dir, epochs, patience, cache, log_every)
        # Released training loop: `rounds` x fit(epochs=round_epochs) with EarlyStopping(patience) and a
        # ModelCheckpoint(save_best_only) re-created every round (its "best" resets), a fresh Adam per
        # round, and from round 2 on: reload that file's weights and multiply lr by round_lr_decay.
        # best.pt therefore ends as the best epoch of the last round, as in the original.
        os.makedirs(ckpt_dir, exist_ok=True)
        self.attach_mask(data["train_ds"])
        train_c, val_c = self.cached_data(data, cache)
        if a.round_schedule:
            return self._fit_round_replay(train_c, ckpt_dir)
        if val_c is None:
            raise SystemExit("schedule upstream_rounds needs a validation split (--val-frac > 0) "
                             "or a fixed --round-schedule")
        loader = self.make_train_loader(train_c)
        net = self.nets["G"]
        best_path = os.path.join(ckpt_dir, "best.pt")
        history, epoch, lr, round_best_state = [], 0, a.lr, None
        self.train_mode(True)
        for r in range(1, a.rounds + 1):
            if r > 1:
                net.load_state_dict(round_best_state)
                lr *= a.round_lr_decay
            self.optimizers["G"] = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            best, wait = float("inf"), 0
            for _ in range(a.round_epochs):
                epoch += 1
                self.epoch = epoch
                te = time.time()
                losses = self.train_epoch(loader, epoch)
                val = self.validate(val_c)
                rec = {"epoch": epoch, "round": r, "val_mse": val, "time_s": time.time() - te, **losses}
                history.append(rec)
                flag = ""
                if val < best:
                    best, wait, flag = val, 0, "  *best"
                    round_best_state = copy.deepcopy(net.state_dict())
                    self.save(best_path, epoch, val, history, extra={"round": r})
                else:
                    wait += 1
                print(f"[{self.name}] round {r}/{a.rounds} epoch {epoch:3d}  lr {lr:.3g}  "
                      f"train_mse {losses['train_mse']:.5g}  val_mse {val:.5f}  ({rec['time_s']:.0f}s){flag}", flush=True)
                with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
                    json.dump(history, f, indent=1)
                if wait >= a.patience:
                    print(f"[{self.name}] round {r}: early stop after {wait} epochs without improvement", flush=True)
                    break
        print(f"[{self.name}] done. best.pt = best epoch of round {a.rounds} (val_mse {best:.5f})", flush=True)
        return history

    def _fit_round_replay(self, train_c, ckpt_dir):
        # Refit on the whole development pool with the round lengths fixed in advance (the 5-fold CV
        # median of where each round's best epoch fell — scripts/cv_round_schedule.py). Each round
        # trains exactly that many epochs, so its last epoch stands in for the round's best epoch
        # and no reload is needed; fresh Adam and lr × round_lr_decay per round as in fit().
        a = self.args
        schedule = [int(n) for n in str(a.round_schedule).split(",") if n.strip()]
        if len(schedule) != a.rounds or min(schedule) < 1:
            raise SystemExit(f"--round-schedule needs {a.rounds} positive epoch counts, got {schedule}")
        loader = self.make_train_loader(train_c)
        net = self.nets["G"]
        history, epoch, lr = [], 0, a.lr
        self.train_mode(True)
        for r, n_ep in enumerate(schedule, start=1):
            if r > 1:
                lr *= a.round_lr_decay
            self.optimizers["G"] = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            for _ in range(n_ep):
                epoch += 1
                self.epoch = epoch
                te = time.time()
                losses = self.train_epoch(loader, epoch)
                rec = {"epoch": epoch, "round": r, "time_s": time.time() - te, **losses}
                history.append(rec)
                print(f"[{self.name}] replay round {r}/{a.rounds} ({n_ep} ep) epoch {epoch:3d}  lr {lr:.3g}  "
                      f"train_mse {losses['train_mse']:.5g}  ({rec['time_s']:.0f}s)", flush=True)
                with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
                    json.dump(history, f, indent=1)
        self.save(os.path.join(ckpt_dir, "best.pt"), epoch, float("nan"), history,
                  extra={"round_schedule": schedule})
        print(f"[{self.name}] done. best.pt = end of the replayed schedule {schedule} ({epoch} epochs)", flush=True)
        return history

    @torch.no_grad()
    def predict(self, mri):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            out = self.nets["G"](mri)
        return out.float() * self.mask
