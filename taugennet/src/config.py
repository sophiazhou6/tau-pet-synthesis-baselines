import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Volume dimensions
# (96, 112, 96) covers the full MNI brain at ~2 mm (2× downsampled from 1 mm native),
# divisible by LATENT_SCALE=8 in every dimension, no cropping.
# Paper used (160, 160, 96) which crops ~47% of the Z axis; kept as a comment
# for reference: VOL_SHAPE = (160, 160, 96)
VOL_SHAPE    = (96, 112, 96)
LATENT_CH    = 3      # was 4 (pre-paper-alignment; paper specifies 3-channel latent)
LATENT_SCALE = 8              # spatial downscale factor of autoencoder

LAT_H = VOL_SHAPE[0] // LATENT_SCALE
LAT_W = VOL_SHAPE[1] // LATENT_SCALE
LAT_D = VOL_SHAPE[2] // LATENT_SCALE

# Diffusion
T_STEPS    = 1000
BETA_START = 0.0015  # was 1e-4; paper uses scaled linear schedule (0.0015 → 0.0205)
BETA_END   = 0.0205  # was 0.02
GROUPNORM_GROUPS = 32  # was hardcoded 8 in models.py; paper specifies 32 groups, ε=1e-6

# Training  (paper: batch=8, epochs=600 on A100)
AE_EPOCHS   = 100
DIFF_EPOCHS = 2000
BATCH_SIZE  = 8
LR          = 1e-4

# Data paths — leave empty to use synthetic data
PET_DIR  = ""   # directory of .nii.gz tau PET volumes
MRI_DIR  = ""   # directory of registered T1 MRI volumes
CSV_PATH = ""   # CSV with columns: subject_id, ptau217

#USE_SYNTHETIC  = PET_DIR == ""
USE_SYNTHETIC  = False
N_SYNTH_TRAIN  = 40
N_SYNTH_TEST   = 10

CLIP_DIM  = 512   # must match CLIP ViT-B/32 hidden size
COND_DIM  = 512   # conditioning embedding dimension (both conditioners output this)
N_REGIONS = 86

# Portable root: set TAUGENNET_ROOT on a new server; defaults to the Princeton path.
ROOT = os.environ.get("TAUGENNET_ROOT", "/scratch/network/sz3962/taugennet")

BASE_DIR          = os.path.join(ROOT, "data/raw")
CLIP_MODEL_PATH   = os.path.join(ROOT, "clip_model")
BIOMEDBERT_MODEL_PATH = os.path.join(ROOT, "biomedbert_model")
ADNI_FLUID_CSV    = os.path.join(ROOT, "data/raw/ADNI34Tau_withFluidBiomarkers.csv")
CHECKPOINT_DIR    = os.path.join(ROOT, "results/checkpoints")
FIGURES_DIR       = os.path.join(ROOT, "results/figures")
GENERATED_DIR     = os.path.join(ROOT, "data/generated")

AE_CHECKPOINT_PATH   = os.path.join(ROOT, "results/checkpoints/taugennet_checkpoint.pt")
DIFF_CHECKPOINT_PATH = os.path.join(ROOT, "results/checkpoints/taugennet_checkpoint_8_epochs.pt")

