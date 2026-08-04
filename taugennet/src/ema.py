"""Exponential moving average (EMA) of model parameters.

Standard diffusion-training stabiliser: keep a shadow copy of the tracked
module's parameters and update it after every optimizer step
    shadow = decay * shadow + (1 - decay) * param.
At validation / sampling / checkpoint time, swap the EMA weights in via
store() -> copy_to() -> (do work) -> restore().

Disabled by passing decay <= 0 (callers simply don't construct an EMA then).
"""
import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def store(self, model):
        """Snapshot current (raw) weights so they can be restored after eval."""
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def copy_to(self, model):
        """Load EMA weights into the model (call store() first to be able to undo)."""
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        """Undo copy_to(): put the raw weights back."""
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._backup:
                p.copy_(self._backup[n])
        self._backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd.get("decay", self.decay)
        self.shadow = sd["shadow"]

    def to(self, device):
        self.shadow = {n: v.to(device) for n, v in self.shadow.items()}
        return self
