"""
Joint cross-modal synthesis + diagnosis (Wang et al., MedIA 2024), three stages as in
the reference scripts (scripts/pretraining_stage/share_gan_*.sh, diag_pretrain.sh,
scripts/joint_stage/joint_learning.sh):

  Stage A  ShareGAN pretraining — UNPAIRED: MRI of subject i vs PET of a random subject j.
           Single generator G(·, alpha) used in both directions (alpha 0: MRI→PET, 1: PET→MRI),
           two discriminators (D_A judges PET, D_B judges MRI), LSGAN; cycle L1 λ_A=λ_B=10,
           identity L1 weight 0.3·λ; Adam(β1 0.5) lr_G = lr_D = 2e-4, exponential decay
           0.95^(epoch-15) after 15 epochs, up to 80 epochs, gradient accumulation over 2
           batches (update_step), image pool 10, early stopping (val, patience 20).
  Stage B  AwareNet diagnosis pretraining on real MRI: CE loss, Adam lr 1e-4 (β1 0.5),
           ExponentialLR γ=0.95 per epoch, batch 2, ≤80 epochs, early stopping patience 10.
  Stage C  Joint training: G (lr 1e-6 = 0.1·lr_G) + classifier (lr 1e-5) in one optimizer,
           D_A/D_B at 1e-6; loss_G = adversarial + cycle + λ_cls·CE where the CE term is
           computed on the real MRI or on the synthesized PET (50/50 per step); lr × 0.7
           per epoch; 15 epochs; batch 1; update_step 2.

Labels: the manifest's diagnosis group (CN/MCI/AD present in the cohort → 2- or 3-class).
Checkpoints of every stage are selected by the common validation criterion (masked MSE
of the MRI→PET synthesis against the paired PET); their code early-stopped on the
validation generator loss (validation PET was paired in their setup too).

Optional supervised term (--paired-l1 λ, default 0 = the published, unpaired method): adds
λ·L1(G(MRI_i,0), PET_i) in stages A and C — our data ARE paired; registered as the model
name 'jointsyn_paired' with λ = 10 (the cycle-loss weight).
"""
import itertools
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseTrainer
from ..cohort import UnpairedView
from ..config import VOL_SHAPE
from ..models.common import GANLoss, ImagePool, from_pm1, init_weights, to_pm1
from ..models.sharesynnet import AwareNet, DefinedDiscriminator, ShareSynNet


class JointSynTrainer(BaseTrainer):
    name = "jointsyn"

    def build(self):
        a = self.args
        n_cls = max(2, len(self.cls_map)) if self.cls_map else 2
        G = ShareSynNet(1, 1)
        DA, DB = DefinedDiscriminator(1), DefinedDiscriminator(1)
        C = AwareNet(num_classes=n_cls, n_slices=getattr(a, 'n_slices', None) or VOL_SHAPE[2])
        for net in (G, DA, DB, C):
            init_weights(net, "kaiming")
        self.nets = {"G": G.to(self.device), "D_A": DA.to(self.device), "D_B": DB.to(self.device),
                     "Cls": C.to(self.device)}
        self.n_cls = n_cls
        self.gan = GANLoss("lsgan")
        self.pool_A, self.pool_B = ImagePool(a.pool_size), ImagePool(a.pool_size)
        self.stage = "A"
        self._set_stage_optimizers("A")

    # ── stage-specific optimizers / schedules ────────────────────────────────
    def _set_stage_optimizers(self, stage):
        a = self.args
        G, DA, DB, C = self.nets["G"], self.nets["D_A"], self.nets["D_B"], self.nets["Cls"]
        self.stage = stage
        if stage == "A":
            self.optimizers = {"G": torch.optim.Adam(G.parameters(), lr=a.lr_g, betas=(a.beta1, 0.999)),
                               "D": torch.optim.Adam(itertools.chain(DA.parameters(), DB.parameters()),
                                                     lr=a.lr_d, betas=(a.beta1, 0.999))}
            f = lambda e: min(1.0, a.lr_gamma_share ** (e - a.n_const_epochs))  # noqa: E731
        elif stage == "B":
            self.optimizers = {"Cls": torch.optim.Adam(C.parameters(), lr=a.lr_cls, betas=(a.beta1, 0.999))}
            f = lambda e: a.lr_gamma_cls ** e  # noqa: E731
        else:
            self.optimizers = {"G": torch.optim.Adam([{"params": G.parameters(), "lr": a.lr_joint_g * 0.1},
                                                      {"params": C.parameters(), "lr": a.lr_joint_g}],
                                                     betas=(a.beta1, 0.999)),
                               "D": torch.optim.Adam(itertools.chain(DA.parameters(), DB.parameters()),
                                                     lr=a.lr_joint_d, betas=(a.beta1, 0.999))}
            f = lambda e: min(1.0, a.lr_gamma_joint ** e)  # noqa: E731
        self.scheds = [torch.optim.lr_scheduler.LambdaLR(o, f) for o in self.optimizers.values()]

    def on_epoch_end(self, epoch):
        for s in self.scheds:
            s.step()

    def make_train_loader(self, cached):
        self._cached_train = cached
        bs = self.args.batch_size if self.stage != "C" else self.args.joint_batch_size
        if self.stage == "B":
            return DataLoader(cached, batch_size=self.args.cls_batch_size, shuffle=True, num_workers=0)
        return DataLoader(UnpairedView(cached, seed=self.args.seed), batch_size=bs, shuffle=True, num_workers=0)

    # ── losses ───────────────────────────────────────────────────────────────
    def _mask_pm1(self, v):
        return v * self.mask + (-1.0) * (1.0 - self.mask)

    def _g_losses(self, real_A, real_B, pet_pair, label, joint):
        a = self.args
        G, DA, DB, C = self.nets["G"], self.nets["D_A"], self.nets["D_B"], self.nets["Cls"]
        fake_B = self._mask_pm1(G(real_A, 0.0))       # MRI → PET
        rec_A = self._mask_pm1(G(fake_B, 1.0))
        fake_A = self._mask_pm1(G(real_B, 1.0))       # PET → MRI
        rec_B = self._mask_pm1(G(fake_A, 0.0))
        losses = {
            "G_A": self.gan(DA(fake_B), True), "G_B": self.gan(DB(fake_A), True),
            "cyc_A": F.l1_loss(rec_A, real_A) * a.lambda_cycle,
            "cyc_B": F.l1_loss(rec_B, real_B) * a.lambda_cycle}
        if a.lambda_identity > 0:
            losses["idt_A"] = F.l1_loss(self._mask_pm1(G(real_B, 0.0)), real_B) * a.lambda_cycle * a.lambda_identity
            losses["idt_B"] = F.l1_loss(self._mask_pm1(G(real_A, 1.0)), real_A) * a.lambda_cycle * a.lambda_identity
        if a.paired_l1 > 0:
            losses["pair_l1"] = F.l1_loss(fake_B, pet_pair) * a.paired_l1
        if joint:
            logits = C(real_A)[0] if random.random() >= 0.5 else C(fake_B)[0]
            losses["cls"] = F.cross_entropy(logits, label) * a.lambda_cls
        return losses, fake_B, fake_A

    def _d_step(self, real_A, real_B, fake_A, fake_B, update_step):
        DA, DB = self.nets["D_A"], self.nets["D_B"]
        fb = self.pool_A.query(fake_B.detach()); fa = self.pool_B.query(fake_A.detach())
        loss = 0.5 * (self.gan(DA(real_B), True) + self.gan(DA(fb), False)) \
             + 0.5 * (self.gan(DB(real_A), True) + self.gan(DB(fa), False))
        (loss / update_step).backward()
        return loss

    def _gan_epoch(self, loader, joint):
        a = self.args
        us = a.update_step
        oG, oD = self.optimizers["G"], self.optimizers["D"]
        sums, n, it = {}, 0, 0
        oG.zero_grad(set_to_none=True); oD.zero_grad(set_to_none=True)
        for mri, pet_j, pet_i, cond in loader:
            real_A, real_B = self._mask_pm1(to_pm1(mri.to(self.device))), self._mask_pm1(to_pm1(pet_j.to(self.device)))
            pet_pair = self._mask_pm1(to_pm1(pet_i.to(self.device)))
            label = cond.view(-1).long().to(self.device)
            for p in itertools.chain(self.nets["D_A"].parameters(), self.nets["D_B"].parameters()):
                p.requires_grad_(False)
            losses, fake_B, fake_A = self._g_losses(real_A, real_B, pet_pair, label, joint)
            (sum(losses.values()) / us).backward()
            for p in itertools.chain(self.nets["D_A"].parameters(), self.nets["D_B"].parameters()):
                p.requires_grad_(True)
            loss_d = self._d_step(real_A, real_B, fake_A, fake_B, us)
            it += 1
            if it % us == 0:
                oG.step(); oG.zero_grad(set_to_none=True)
                oD.step(); oD.zero_grad(set_to_none=True)
            b = mri.shape[0]
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item() * b
            sums["D"] = sums.get("D", 0.0) + loss_d.item() * b
            n += b
        if it % us != 0:
            oG.step(); oG.zero_grad(set_to_none=True); oD.step(); oD.zero_grad(set_to_none=True)
        out = {k: v / max(n, 1) for k, v in sums.items()}
        out["lr_G"] = oG.param_groups[0]["lr"]
        return out

    def _cls_epoch(self, loader):
        C, opt = self.nets["Cls"], self.optimizers["Cls"]
        tot, correct, n = 0.0, 0, 0
        for pet, mri, cond in loader:
            x = self._mask_pm1(to_pm1(mri.to(self.device)))
            y = cond.view(-1).long().to(self.device)
            opt.zero_grad(set_to_none=True)
            logits = C(x)[0]
            loss = F.cross_entropy(logits, y)
            loss.backward(); opt.step()
            tot += loss.item() * y.numel(); correct += (logits.argmax(1) == y).sum().item(); n += y.numel()
        return {"cls_ce": tot / max(n, 1), "cls_acc": correct / max(n, 1), "lr_cls": opt.param_groups[0]["lr"]}

    def train_epoch(self, loader, epoch):
        if self.stage == "B":
            return self._cls_epoch(loader)
        return self._gan_epoch(loader, joint=(self.stage == "C"))

    @torch.no_grad()
    def predict(self, mri):
        x = self._mask_pm1(to_pm1(mri))
        return from_pm1(self._mask_pm1(self.nets["G"](x, 0.0))) * self.mask

    @torch.no_grad()
    def classify(self, cached, on="mri"):
        """Diagnosis metrics of the classifier on real MRI ('mri') or synthesized PET ('synpet')."""
        C = self.nets["Cls"]
        self.train_mode(False)
        ys, ps = [], []
        for pet, mri, cond in DataLoader(cached, batch_size=2, shuffle=False, num_workers=0):
            x = self._mask_pm1(to_pm1(mri.to(self.device)))
            if on == "synpet":
                x = self._mask_pm1(self.nets["G"](x, 0.0))
            ps.append(torch.softmax(C(x)[0], 1).cpu()); ys.append(cond.view(-1).long())
        self.train_mode(True)
        y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
        out = {f"acc_{on}": float((p.argmax(1) == y).mean())}
        if self.n_cls == 2 and len(set(y.tolist())) == 2:
            from sklearn.metrics import roc_auc_score
            out[f"auc_{on}"] = float(roc_auc_score(y, p[:, 1]))
        return out

    def extra_val_metrics(self, val_cached):
        if self.stage == "A" or val_cached is None:
            return {}
        return {**self.classify(val_cached, "mri"), **self.classify(val_cached, "synpet")}

    # ── three-stage fit ──────────────────────────────────────────────────────
    def fit(self, data, ckpt_dir, epochs=None, patience=None, cache=True, log_every=1):
        import os
        a = self.args
        hist = {}
        # Stage A: ShareGAN pretraining (unpaired)
        self._set_stage_optimizers("A")
        hist["A"] = super().fit(data, os.path.join(ckpt_dir, "stageA"), epochs=a.epochs_share,
                                patience=a.patience_share, cache=cache, log_every=log_every)
        self.load_state_dict(torch.load(os.path.join(ckpt_dir, "stageA", "best.pt"), map_location=self.device,
                                        weights_only=False)["state"])
        # Stage B: classifier pretraining on real MRI (paired loader, labels)
        self._set_stage_optimizers("B")
        self._validate_by = "cls"
        hist["B"] = self._fit_cls(data, os.path.join(ckpt_dir, "stageB"), cache=cache)
        # Stage C: joint
        self._set_stage_optimizers("C")
        hist["C"] = super().fit(data, os.path.join(ckpt_dir, "stageC"), epochs=a.epochs_joint,
                                patience=0, cache=cache, log_every=log_every)
        best = torch.load(os.path.join(ckpt_dir, "stageC", "best.pt"), map_location=self.device, weights_only=False)
        self.load_state_dict(best["state"])
        self.save(os.path.join(ckpt_dir, "best.pt"), best["epoch"], best["val_mse"], hist)
        os.remove(os.path.join(ckpt_dir, "stageC", "best.pt"))      # identical to best.pt — don't keep two copies
        for f in ("stageA/last.pt", "stageC/last.pt"):
            if os.path.exists(os.path.join(ckpt_dir, f)):
                os.remove(os.path.join(ckpt_dir, f))
        return hist

    def _fit_cls(self, data, ckpt_dir, cache=True):
        """Stage B loop: select by validation cross-entropy (patience_cls)."""
        import os, json, time
        a = self.args
        os.makedirs(ckpt_dir, exist_ok=True)
        train_c, val_c = self.cached_data(data, cache)
        loader = self.make_train_loader(train_c)
        best, bad, history = float("inf"), 0, []
        C = self.nets["Cls"]
        for epoch in range(1, a.epochs_cls + 1):
            t0 = time.time()
            C.train(True)
            losses = self._cls_epoch(loader)
            self.on_epoch_end(epoch)
            val_ce = float("nan"); extra = {}
            if val_c is not None:
                C.train(False)
                tot, n = 0.0, 0
                with torch.no_grad():
                    for pet, mri, cond in DataLoader(val_c, batch_size=2, shuffle=False):
                        x = self._mask_pm1(to_pm1(mri.to(self.device))); y = cond.view(-1).long().to(self.device)
                        tot += F.cross_entropy(C(x)[0], y, reduction="sum").item(); n += y.numel()
                val_ce = tot / max(n, 1)
                extra = self.classify(val_c, "mri")
            rec = {"epoch": epoch, "val_ce": val_ce, "time_s": time.time() - t0, **losses, **extra}
            history.append(rec)
            flag = ""
            if val_c is None or val_ce < best:
                best, bad, flag = val_ce, 0, "  *best"
                torch.save({"state": {"Cls": C.state_dict()}, "epoch": epoch, "val_ce": val_ce},
                           os.path.join(ckpt_dir, "best.pt"))     # classifier only (the other nets are unchanged)
            else:
                bad += 1
            print(f"[jointsyn/cls] epoch {epoch:3d}/{a.epochs_cls}  " + "  ".join(f"{k} {v:.4g}" for k, v in losses.items())
                  + f"  val_ce {val_ce:.4f}  " + "  ".join(f"{k} {v:.3f}" for k, v in extra.items()) + flag, flush=True)
            with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=1)
            if val_c is not None and a.patience_cls and bad >= a.patience_cls:
                print("[jointsyn/cls] early stopping."); break
        C.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=self.device, weights_only=False)["state"]["Cls"])
        return history
