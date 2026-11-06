#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""generate_posterior_mean.py — diffusion POSTERIOR MEAN via K DDPM samples per subject.

Tests whether single-sample noise (grainy glass brain, low voxel Pearson) is hiding a good
underlying prediction: averaging K stochastic samples -> the model's conditional-mean estimate,
directly comparable to a deterministic regressor (DenseUNet/INR). Saves the mean as
subject_{i}.npy (normalized [0,1]) for evaluate_final --use-cached.

dataset_final path (atrophy / ptau / combined via MLP conditioner). Legacy split.
"""
import argparse, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from src.config import DEVICE
from src.inference import load_models, synthesize_tau_pet
from src.diffusion import DiffusionSchedule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--use-best", action="store_true")
    p.add_argument("--unet-channels", default="128,256,512")
    p.add_argument("--n-transformer", type=int, default=1)
    p.add_argument("--arch", default="silu")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--k", type=int, default=16, help="samples per subject to average")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--use-mentor-split", action=argparse.BooleanOptionalAction, default=True,
                   help="MUST match the split the checkpoint was trained on. Legacy checkpoints "
                        "were trained pre-mentor-split -> pass --no-use-mentor-split for those.")
    p.add_argument("--use-mask", action=argparse.BooleanOptionalAction, default=True,
                   help="DK86 mask (default on).")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    suffix = "_best" if a.use_best else ""
    ckpt = os.path.join(a.checkpoint_dir, f"diff_{a.mode}{suffix}.pt")
    ch = tuple(int(x) for x in a.unet_channels.split(","))
    ae, unet, conditioner, latent_std, _ = load_models(
        ckpt, a.mode, arch=a.arch, ch_list=ch, n_transformer=a.n_transformer)
    encode_cond = conditioner.encode
    # The sampler's beta schedule MUST match the one the checkpoint was trained with, or
    # samples are silently corrupted. Legacy checkpoints have no field -> linear.
    _ck = torch.load(ckpt, map_location="cpu")
    _ns = _ck.get("noise_schedule", "linear")
    del _ck
    print(f"Noise schedule (from checkpoint): {_ns}", flush=True)
    schedule = DiffusionSchedule(schedule=_ns)

    fk = {} if a.fold is None else dict(fold_idx=a.fold, n_folds=a.n_folds)
    # Pick the dataset that matches the conditioning mode. 'combined' (atrophy MLP + CLIP p-tau217)
    # lives in dataset_combined; ptau217 / ptau217_mlp / atrophy live in dataset_final.
    if a.mode == "combined":
        from src import dataset_combined as D
    else:
        from src import dataset_final as D
    test_ds = D.build_dataloaders(a.mode, use_dk_mask=a.use_mask,
                                  use_mentor_split=a.use_mentor_split, **fk)[2]
    print(f"test subjects={len(test_ds)}  K={a.k}  steps={a.n_steps}  "
          f"mentor_split={a.use_mentor_split}  dk86={a.use_mask}", flush=True)

    for i in range(len(test_ds)):
        pet, mri, cond = test_ds[i]
        acc = None
        with torch.no_grad():
            for _ in range(a.k):
                g = synthesize_tau_pet(
                    mri.unsqueeze(0).to(DEVICE), cond.unsqueeze(0).to(DEVICE),
                    ae, unet, schedule, encode_cond, latent_std,
                    n_steps=a.n_steps, sampler="ddpm").squeeze().cpu().numpy()
                acc = g if acc is None else acc + g
        np.save(os.path.join(a.out_dir, f"subject_{i:03d}.npy"), (acc / a.k).astype(np.float32))
        if i % 5 == 0:
            print(f"  subject {i}/{len(test_ds)} (mean of {a.k})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
