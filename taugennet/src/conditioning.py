"""
conditioning.py – interchangeable conditioning strategies for TauGenNet.

All conditioners expose the same interface:
    conditioner.encode(data)  → (B, n_tokens, COND_DIM)
    conditioner.out_dim       → int  (= COND_DIM = 512)
    conditioner.trainable_parameters() → list[Parameter]
    conditioner.state_dict() / load_state_dict()
    conditioner.train() / eval()

Available modes (build_conditioner factory):
    ptau217          – frozen CLIP encodes "Plasma is X.XXX." (no trainable params)
    ptau217_mlp      – learned MLP for scalar p-tau217
    atrophy          – learned MLP for 86-dim atrophy z-score vector
    combined         – atrophy MLP + CLIP p-tau217;  cond dim = 87
    combined_tissue  – combined + tissue-type summary; cond dim = 90
    combined_demo    – combined + demographic CLIP;   cond dim = 91
    combined_full    – combined + tissue + demo;      cond dim = 94
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer

from .config import BIOMEDBERT_MODEL_PATH, CLIP_MODEL_PATH, COND_DIM, DEVICE


# ── Tissue-class constants ────────────────────────────────────────────────────
#
# DK atlas region index → tissue class (0-indexed positions within the 86-dim vector):
#   0 = cortical GM   (indices 0-67, all CTX_LH/RH_* regions)
#   1 = subcortical GM (indices 69-76, 78-85: thalamus, caudate, putamen, pallidum,
#                       hippocampus, amygdala, accumbens, ventralDC × both hemispheres)
#   2 = cerebellar     (indices 68, 77: LEFT/RIGHT_CEREBELLUM_CORTEX)
_TISSUE_CLASS = np.zeros(86, dtype=np.int64)
_TISSUE_CLASS[68]    = 2  # LEFT_CEREBELLUM_CORTEX
_TISSUE_CLASS[77]    = 2  # RIGHT_CEREBELLUM_CORTEX
_TISSUE_CLASS[69:77] = 1  # left subcortical
_TISSUE_CLASS[78:86] = 1  # right subcortical

# ADNI DIAGNOSIS codes: 1=CN, 2=MCI, 3=AD
_DIAG_LABEL = {1: "cognitively normal", 2: "mild cognitive impairment",
               3: "Alzheimer's disease"}
_GENDER_LABEL = {1: "male", 2: "female"}


# ── p-tau217 conditioner ──────────────────────────────────────────────────────

class PTau217Conditioner:
    """Frozen CLIP text encoder conditioner for scalar plasma p-tau217."""

    out_dim = COND_DIM

    def __init__(self, model_path=CLIP_MODEL_PATH, device=DEVICE):
        self.device    = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.model     = CLIPTextModel.from_pretrained(model_path).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # CLIP hidden size is 512 for ViT-B/32; confirm it matches COND_DIM
        assert self.model.config.hidden_size == COND_DIM, (
            f"CLIP hidden size {self.model.config.hidden_size} ≠ COND_DIM {COND_DIM}"
        )

    @torch.no_grad()
    def encode(self, ptau_vals):
        """
        ptau_vals : (B,) or (B, 1) float tensor of p-tau217 values
        Returns   : (B, seq_len, COND_DIM)
        """
        if ptau_vals.dim() == 2:
            ptau_vals = ptau_vals.squeeze(1)
        prompts = [f"Plasma is {v.item():.3f}." for v in ptau_vals]
        tokens  = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        return self.model(**tokens).last_hidden_state  # (B, seq_len, 512)

    def trainable_parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        pass  # frozen; nothing to load

    def train(self): pass
    def eval(self):  pass

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self


class PTau217BioBERTConditioner(nn.Module):
    """Frozen PubMedBERT (BiomedBERT) text encoder for scalar plasma p-tau217.

    Drop-in alternative to PTau217Conditioner: verbalises the biomarker as a
    biomedical-domain prompt and encodes it with a frozen BERT encoder. BERT-base
    hidden size is 768 ≠ COND_DIM (512), so a small learned linear projection maps
    the token embeddings to COND_DIM. The projection is the only trainable head.
    """

    out_dim = COND_DIM

    def __init__(self, model_path=BIOMEDBERT_MODEL_PATH, device=DEVICE):
        super().__init__()
        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        hidden = self.model.config.hidden_size  # 768 for PubMedBERT-base
        self.proj = nn.Linear(hidden, COND_DIM).to(device)

    def encode(self, ptau_vals):
        """
        ptau_vals : (B,) or (B, 1) float tensor of p-tau217 values
        Returns   : (B, seq_len, COND_DIM)
        """
        if ptau_vals.dim() == 2:
            ptau_vals = ptau_vals.squeeze(1)
        prompts = [f"Plasma p-tau217 level {v.item():.3f} pg/mL." for v in ptau_vals]
        tokens  = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**tokens).last_hidden_state  # (B, seq_len, 768)
        return self.proj(hidden)  # (B, seq_len, 512)

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.proj.parameters())

    def state_dict(self, *args, **kwargs):
        # persist only the learned projection; the BERT encoder is frozen/on disk
        return self.proj.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.proj.load_state_dict(state_dict, strict=strict)

    def to(self, device):
        self.model = self.model.to(device)
        self.proj  = self.proj.to(device)
        self.device = device
        return self


# ── atrophy conditioner ───────────────────────────────────────────────────────

class AtrophyConditioner(nn.Module):
    """Learned MLP conditioner for 86-dim regional atrophy z-score vectors.

    Produces a sequence of n_tokens context vectors for cross-attention,
    using all 86 regions rather than the 20-region CLIP-text truncation.
    """

    out_dim = COND_DIM

    def __init__(self, n_regions=86, out_dim=COND_DIM, n_tokens=16):
        super().__init__()
        self.n_tokens = n_tokens
        hidden = 256
        self.mlp = nn.Sequential(
            nn.LayerNorm(n_regions),
            nn.Linear(n_regions, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim * n_tokens),
        )

    def encode(self, atrophy_vals):
        """
        atrophy_vals : (B, 86) float tensor of regional atrophy z-scores
        Returns      : (B, n_tokens, COND_DIM)
        """
        return self.mlp(atrophy_vals).view(
            atrophy_vals.shape[0], self.n_tokens, self.out_dim
        )

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.parameters())


# ── null conditioner ──────────────────────────────────────────────────────────

class NullConditioner(nn.Module):
    """Learned null context — for spatial-map-only conditioning (no biomarker
    cross-attention). Returns a single learned token broadcast to the batch, so
    the UNet cross-attention has a constant (learned) context while all
    subject-specific conditioning flows through the spatial map channel(s).
    The input is ignored except for its batch dimension.
    """

    out_dim = COND_DIM

    def __init__(self, out_dim=COND_DIM, n_tokens=1):
        super().__init__()
        self.n_tokens = n_tokens
        self.null = nn.Parameter(torch.zeros(1, n_tokens, out_dim))

    def encode(self, x):
        return self.null.expand(x.shape[0], -1, -1)

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.parameters())


# ── p-tau217 MLP conditioner ─────────────────────────────────────────────────

class PTau217MLPConditioner(nn.Module):
    """Learned MLP conditioner for scalar plasma p-tau217.

    Mirrors AtrophyConditioner but for a 1-dim input. Uses hidden-layer
    LayerNorm instead of input LayerNorm (LayerNorm of a single value
    collapses to zero).
    """

    out_dim = COND_DIM

    def __init__(self, out_dim=COND_DIM, n_tokens=16):
        super().__init__()
        self.n_tokens = n_tokens
        hidden = 256
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim * n_tokens),
        )

    def encode(self, ptau_vals):
        """ptau_vals : (B,) or (B, 1) float tensor; returns (B, n_tokens, COND_DIM)."""
        if ptau_vals.dim() == 1:
            ptau_vals = ptau_vals.unsqueeze(1)
        return self.mlp(ptau_vals).view(ptau_vals.shape[0], self.n_tokens, self.out_dim)

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.parameters())


# ── combined conditioner ─────────────────────────────────────────────────────

class CombinedConditioner(nn.Module):
    """Encodes (B, 87) = [atrophy(86) ‖ ptau217(1)] into cross-attention tokens.

    Atrophy → AtrophyConditioner MLP  → (B, 16, 512)
    ptau217 → PTau217Conditioner CLIP → (B, seq, 512)
    Output  → concatenated along dim=1 → (B, 16+seq, 512)

    Only the atrophy MLP has trainable parameters; CLIP stays frozen.
    """

    out_dim = COND_DIM

    def __init__(self, device=DEVICE):
        super().__init__()
        self.atrophy = AtrophyConditioner().to(device)
        self.ptau    = PTau217Conditioner(device=device)

    def encode(self, cond):
        """cond : (B, 87) — first 86 dims = atrophy, last 1 = ptau217."""
        atrophy_tokens = self.atrophy.encode(cond[:, :86])  # (B, 16, 512)
        ptau_tokens    = self.ptau.encode(cond[:, 86:])     # (B, seq, 512)
        return torch.cat([atrophy_tokens, ptau_tokens], dim=1)

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.atrophy.parameters())

    def state_dict(self):
        return {"atrophy": self.atrophy.state_dict()}

    def load_state_dict(self, sd, strict=True):
        self.atrophy.load_state_dict(sd["atrophy"])

    def train(self, mode=True):
        self.atrophy.train(mode)
        return self

    def eval(self):
        self.atrophy.eval()
        return self

    def to(self, device):
        self.atrophy = self.atrophy.to(device)
        self.ptau    = self.ptau.to(device)
        return self


class CombinedMLPConditioner(nn.Module):
    """Like CombinedConditioner but ptau217 uses a learned MLP instead of frozen CLIP.

    Atrophy → AtrophyConditioner MLP    → (B, 16, 512)
    ptau217 → PTau217MLPConditioner MLP → (B, n, 512)
    Output  → concatenated along dim=1

    BOTH branches are trainable here (no frozen CLIP), so state_dict carries both.
    """

    out_dim = COND_DIM

    def __init__(self, device=DEVICE):
        super().__init__()
        self.atrophy = AtrophyConditioner().to(device)
        self.ptau    = PTau217MLPConditioner().to(device)

    def encode(self, cond):
        """cond : (B, 87) — first 86 dims = atrophy, last 1 = ptau217."""
        atrophy_tokens = self.atrophy.encode(cond[:, :86])
        ptau_tokens    = self.ptau.encode(cond[:, 86:])
        return torch.cat([atrophy_tokens, ptau_tokens], dim=1)

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.atrophy.parameters()) + list(self.ptau.parameters())

    def state_dict(self):
        return {"atrophy": self.atrophy.state_dict(), "ptau": self.ptau.state_dict()}

    def load_state_dict(self, sd, strict=True):
        self.atrophy.load_state_dict(sd["atrophy"])
        self.ptau.load_state_dict(sd["ptau"])

    def train(self, mode=True):
        self.atrophy.train(mode); self.ptau.train(mode); return self

    def eval(self):
        self.atrophy.eval(); self.ptau.eval(); return self

    def to(self, device):
        self.atrophy = self.atrophy.to(device); self.ptau = self.ptau.to(device); return self


# ── tissue-type conditioner ──────────────────────────────────────────────────

class TissueTypeConditioner(nn.Module):
    """Encodes per-tissue-class mean atrophy (3 classes) into cross-attention tokens.

    Input:  (B, 3) = [mean_cortical, mean_subcortical, mean_cerebellar] atrophy z-scores
    Output: (B, n_tokens=4, COND_DIM=512)
    """

    out_dim = COND_DIM

    def __init__(self, out_dim=COND_DIM, n_tokens=4):
        super().__init__()
        self.n_tokens = n_tokens
        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, out_dim * n_tokens),
        )

    def encode(self, tissue_vals):
        """tissue_vals: (B, 3) → (B, n_tokens, COND_DIM)."""
        return self.mlp(tissue_vals).view(tissue_vals.shape[0], self.n_tokens, self.out_dim)

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.parameters())


class CombinedWithTissueConditioner(nn.Module):
    """Combined conditioning with tissue-type summary appended.

    Input:  (B, 90) = [atrophy(86) | ptau217(1) | tissue_summary(3)]
    Output: (B, 16 + seq + 4, 512) concatenated tokens

    Trainable: atrophy MLP + tissue MLP; CLIP stays frozen.
    """

    out_dim = COND_DIM

    def __init__(self, device=DEVICE):
        super().__init__()
        self.atrophy = AtrophyConditioner().to(device)
        self.ptau    = PTau217Conditioner(device=device)
        self.tissue  = TissueTypeConditioner().to(device)

    def encode(self, cond):
        """cond: (B, 90) = [atrophy(86) | ptau(1) | tissue(3)]"""
        atrophy_tokens = self.atrophy.encode(cond[:, :86])   # (B, 16, 512)
        ptau_tokens    = self.ptau.encode(cond[:, 86:87])    # (B, seq, 512)
        tissue_tokens  = self.tissue.encode(cond[:, 87:90])  # (B,  4, 512)
        return torch.cat([atrophy_tokens, ptau_tokens, tissue_tokens], dim=1)

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.atrophy.parameters()) + list(self.tissue.parameters())

    def state_dict(self):
        return {"atrophy": self.atrophy.state_dict(), "tissue": self.tissue.state_dict()}

    def load_state_dict(self, sd, strict=True):
        self.atrophy.load_state_dict(sd["atrophy"])
        self.tissue.load_state_dict(sd["tissue"])

    def train(self, mode=True):
        self.atrophy.train(mode)
        self.tissue.train(mode)
        return self

    def eval(self):
        self.atrophy.eval()
        self.tissue.eval()
        return self

    def to(self, device):
        self.atrophy = self.atrophy.to(device)
        self.ptau    = self.ptau.to(device)
        self.tissue  = self.tissue.to(device)
        return self


# ── demographic CLIP conditioner ─────────────────────────────────────────────

class DemographicCLIPConditioner:
    """Frozen CLIP text encoder for demographic conditioning.

    Formats per-subject scalars (age, gender, diagnosis, education) into a
    natural language prompt and encodes via frozen CLIP ViT-B/32.

    Input:  (B, 4) = [age, gender_code, diag_code, edu_years]
              gender_code: 1=male, 2=female
              diag_code:   1=CN, 2=MCI, 3=AD
    Output: (B, seq_len, 512)
    """

    out_dim = COND_DIM

    def __init__(self, model_path=CLIP_MODEL_PATH, device=DEVICE):
        self.device    = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.model     = CLIPTextModel.from_pretrained(model_path).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, demo_vals):
        """demo_vals: (B, 4) = [age, gender, diag, edu] → (B, seq_len, 512)."""
        prompts = []
        for row in demo_vals:
            age, gender, diag, edu = (
                float(row[0]), int(row[1].item()), int(row[2].item()), float(row[3])
            )
            gender_str = _GENDER_LABEL.get(gender, "unknown gender")
            diag_str   = _DIAG_LABEL.get(diag, "unknown diagnosis")
            prompts.append(
                f"Age {age:.0f}, {gender_str}. Diagnosis: {diag_str}. "
                f"Education: {edu:.0f} years."
            )
        tokens = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        return self.model(**tokens).last_hidden_state  # (B, seq_len, 512)

    def trainable_parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, sd, strict=True):
        pass

    def train(self): pass
    def eval(self):  pass

    def to(self, device):
        self.model  = self.model.to(device)
        self.device = device
        return self


class CombinedWithDemoConditioner(nn.Module):
    """Combined conditioning with demographic CLIP tokens appended.

    Input:  (B, 91) = [atrophy(86) | ptau217(1) | age(1) | gender(1) | diag(1) | edu(1)]
    Output: (B, 16 + seq_ptau + seq_demo, 512) concatenated tokens

    Trainable: atrophy MLP only; both CLIP encoders frozen.
    """

    out_dim = COND_DIM

    def __init__(self, device=DEVICE):
        super().__init__()
        self.atrophy = AtrophyConditioner().to(device)
        self.ptau    = PTau217Conditioner(device=device)
        self.demo    = DemographicCLIPConditioner(device=device)

    def encode(self, cond):
        """cond: (B, 91) = [atrophy(86) | ptau(1) | age | gender | diag | edu]"""
        atrophy_tokens = self.atrophy.encode(cond[:, :86])   # (B, 16, 512)
        ptau_tokens    = self.ptau.encode(cond[:, 86:87])    # (B, seq, 512)
        demo_tokens    = self.demo.encode(cond[:, 87:91])    # (B, seq, 512)
        return torch.cat([atrophy_tokens, ptau_tokens, demo_tokens], dim=1)

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.atrophy.parameters())

    def state_dict(self):
        return {"atrophy": self.atrophy.state_dict()}

    def load_state_dict(self, sd, strict=True):
        self.atrophy.load_state_dict(sd["atrophy"])

    def train(self, mode=True):
        self.atrophy.train(mode)
        return self

    def eval(self):
        self.atrophy.eval()
        return self

    def to(self, device):
        self.atrophy = self.atrophy.to(device)
        self.ptau    = self.ptau.to(device)
        self.demo    = self.demo.to(device)
        return self


class CombinedFullConditioner(nn.Module):
    """Combined conditioning with both tissue-type and demographic tokens.

    Input:  (B, 94) = [atrophy(86) | ptau(1) | tissue(3) | age(1) | gender(1) | diag(1) | edu(1)]
    Output: (B, 16 + seq_ptau + 4 + seq_demo, 512) concatenated tokens

    Trainable: atrophy MLP + tissue MLP; both CLIP encoders frozen.
    """

    out_dim = COND_DIM

    def __init__(self, device=DEVICE):
        super().__init__()
        self.atrophy = AtrophyConditioner().to(device)
        self.ptau    = PTau217Conditioner(device=device)
        self.tissue  = TissueTypeConditioner().to(device)
        self.demo    = DemographicCLIPConditioner(device=device)

    def encode(self, cond):
        """cond: (B, 94) = [atrophy(86)|ptau(1)|tissue(3)|age|gender|diag|edu]"""
        atrophy_tokens = self.atrophy.encode(cond[:, :86])    # (B, 16, 512)
        ptau_tokens    = self.ptau.encode(cond[:, 86:87])     # (B, seq, 512)
        tissue_tokens  = self.tissue.encode(cond[:, 87:90])   # (B,  4, 512)
        demo_tokens    = self.demo.encode(cond[:, 90:94])     # (B, seq, 512)
        return torch.cat([atrophy_tokens, ptau_tokens, tissue_tokens, demo_tokens], dim=1)

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.atrophy.parameters()) + list(self.tissue.parameters())

    def state_dict(self):
        return {"atrophy": self.atrophy.state_dict(), "tissue": self.tissue.state_dict()}

    def load_state_dict(self, sd, strict=True):
        self.atrophy.load_state_dict(sd["atrophy"])
        self.tissue.load_state_dict(sd["tissue"])

    def train(self, mode=True):
        self.atrophy.train(mode)
        self.tissue.train(mode)
        return self

    def eval(self):
        self.atrophy.eval()
        self.tissue.eval()
        return self

    def to(self, device):
        self.atrophy = self.atrophy.to(device)
        self.ptau    = self.ptau.to(device)
        self.tissue  = self.tissue.to(device)
        self.demo    = self.demo.to(device)
        return self


# ── factory ───────────────────────────────────────────────────────────────────

def build_conditioner(mode: str, device=DEVICE):
    """
    mode : 'ptau217' | 'ptau217_mlp' | 'atrophy' | 'combined' |
           'combined_tissue' | 'combined_demo' | 'combined_full'
    Returns an initialised conditioner on the given device.

    Conditioning vector dims by mode:
        combined         → (B, 87)  = [atrophy(86) | ptau(1)]
        combined_tissue  → (B, 90)  = [atrophy(86) | ptau(1) | tissue(3)]
        combined_demo    → (B, 91)  = [atrophy(86) | ptau(1) | age|gender|diag|edu]
        combined_full    → (B, 94)  = [atrophy(86) | ptau(1) | tissue(3) | age|gender|diag|edu]
    """
    if mode == "ptau217":
        return PTau217Conditioner(device=device)
    if mode == "ptau217_biobert":
        return PTau217BioBERTConditioner(device=device)
    if mode == "ptau217_mlp":
        return PTau217MLPConditioner().to(device)
    if mode == "atrophy":
        return AtrophyConditioner().to(device)
    if mode == "combined":
        return CombinedConditioner(device=device)
    if mode == "combined_mlp":
        return CombinedMLPConditioner(device=device)
    if mode == "combined_tissue":
        return CombinedWithTissueConditioner(device=device)
    if mode == "combined_demo":
        return CombinedWithDemoConditioner(device=device)
    if mode == "combined_full":
        return CombinedFullConditioner(device=device)
    raise ValueError(
        f"Unknown conditioning mode {mode!r}. "
        "Use 'ptau217', 'ptau217_biobert', 'ptau217_mlp', 'atrophy', 'combined', "
        "'combined_tissue', 'combined_demo', or 'combined_full'."
    )
