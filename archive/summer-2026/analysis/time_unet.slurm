#!/bin/bash
#SBATCH --job-name=time_unet
#SBATCH --partition=gpu
#SBATCH --qos=hpc_core
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
export TAUGENNET_ROOT=$HOME/taugennet
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
cd $HOME/taugennet
/usr/bin/python3.9 - <<'PY'
import time, torch
from src.config import VOL_SHAPE, LATENT_SCALE
from src.models_spatial import DenoisingUNet3D
unet = DenoisingUNet3D(latent_ch=2, ch_list=(128,256,512), n_transformer=1).cuda().eval()
ch   = unet.in_conv.weight.shape[1]
lat  = torch.randn(1, ch, *[d//LATENT_SCALE for d in VOL_SHAPE], device="cuda")
t    = torch.zeros(1, dtype=torch.long, device="cuda")
ctx  = torch.randn(1, 1, 512, device="cuda")
with torch.no_grad():
    for _ in range(3): unet(lat, t, ctx)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20): unet(lat, t, ctx)
    torch.cuda.synchronize()
per = (time.time()-t0)/20
print("U-Net forward: %.1f ms  ->  500-step volume: %.1f s" % (per*1000, per*500))
print("peak GPU mem: %.2f GB" % (torch.cuda.max_memory_allocated()/1e9))
PY
