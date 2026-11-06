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
# Using original size (96, 112, 96) with ReLU activation for memory efficiency
# Can be changed to (160, 160, 96) with larger GPU or gradient checkpointing
VOL_SHAPE    = (160, 160, 96)
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

# Training  (paper: batch=8 on A100; reduced to 4 for V100 with 160x160x96 volumes)
AE_EPOCHS   = 100
DIFF_EPOCHS = 2000
BATCH_SIZE  = 1
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

BASE_DIR          = "/scratch/network/sz3962/taugennet/data/raw"
CLIP_MODEL_PATH   = "/scratch/network/sz3962/taugennet/clip_model"
ADNI_FLUID_CSV    = "/scratch/network/sz3962/taugennet/data/raw/ADNI34Tau_withFluidBiomarkers.csv"
CHECKPOINT_DIR    = "/scratch/network/sz3962/taugennet/results/checkpoints"
FIGURES_DIR       = "/scratch/network/sz3962/taugennet/results/figures"
GENERATED_DIR     = "/scratch/network/sz3962/taugennet/data/generated"

AE_CHECKPOINT_PATH   = "/scratch/network/sz3962/taugennet/results/checkpoints/taugennet_checkpoint_relu_v160.pt"
DIFF_CHECKPOINT_PATH = "/scratch/network/sz3962/taugennet/results/checkpoints/taugennet_checkpoint_relu_epochs.pt"
