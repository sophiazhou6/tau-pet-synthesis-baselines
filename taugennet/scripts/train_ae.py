import os, time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

from src.config import DEVICE, AE_EPOCHS, LR, AE_CHECKPOINT_PATH
from src.dataset_final import build_dataloaders
from src.models import Autoencoder3D


def ae_loss(recon, x, mean, logvar, kl_weight=1e-4):
    recon_loss = F.l1_loss(recon, x)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss


def train_autoencoder(ae, train_loader, ae_epochs=AE_EPOCHS, lr=LR,
                      device=DEVICE, checkpoint_path=AE_CHECKPOINT_PATH):
    # trains the shared 3D vae on both pet and mri volumes
    # goal: learn a compact latent rep of brain volumes that the diff model will later operate on
    # mixed precision uses float16 for forward/backward pass to halve memory usage
    ae_opt    = torch.optim.Adam(ae.parameters(), lr=lr)
    ae_losses = []
    scaler    = GradScaler()

    training_start = time.time()
    print("=== Autoencoder Pretraining ===", flush=True)

    for epoch in range(ae_epochs):
        epoch_start = time.time()
        ae.train()
        epoch_loss = 0

        for pet, mri, _ in train_loader:
            pet, mri = pet.to(device), mri.to(device)
            ae_opt.zero_grad()
            loss = 0
            with autocast():
                for vol in [pet, mri]:
                    recon, mean, logvar = ae(vol)
                    loss = loss + ae_loss(recon, vol, mean, logvar)
            scaler.scale(loss).backward()
            scaler.step(ae_opt)
            scaler.update()
            epoch_loss += loss.item()

        avg        = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        elapsed    = time.time() - training_start
        remaining  = epoch_time * (ae_epochs - epoch - 1)
        ae_losses.append(avg)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{ae_epochs}  loss={avg:.4f}  "
                  f"epoch={epoch_time:.1f}s  "
                  f"elapsed={elapsed/60:.1f}m  "
                  f"remaining={remaining/60:.1f}m")
            torch.save({
                "ae":        ae.state_dict(),
                "ae_losses": ae_losses,
            }, checkpoint_path)
            print(f"  Checkpoint saved at epoch {epoch+1}")

    total_time = time.time() - training_start
    plt.figure(figsize=(7, 3))
    plt.plot(ae_losses); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Autoencoder Training Loss"); plt.tight_layout(); plt.show()
    print(f"Autoencoder pretraining complete. Total time: {total_time/60:.1f} minutes")

    return ae, ae_losses


if __name__ == "__main__":
    # AE pretraining doesn't use conditioning; 'atrophy' mode gives largest matched set (243 subjects)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(mode="atrophy")

    ae = Autoencoder3D().to(DEVICE)
    print(f"Autoencoder parameters: {sum(p.numel() for p in ae.parameters()):,}")

    # Load pretrained AE checkpoint if available, otherwise train from scratch
    if os.path.exists(AE_CHECKPOINT_PATH):
        ckpt      = torch.load(AE_CHECKPOINT_PATH, map_location=DEVICE)
        ae.load_state_dict(ckpt["ae"])
        ae_losses = ckpt["ae_losses"]
        ae.eval()
        print(f"AE checkpoint loaded. Trained for {len(ae_losses)} epochs, "
              f"final loss={ae_losses[-1]:.4f}")
    else:
        ae, ae_losses = train_autoencoder(ae, train_loader)

    # ── AE Reconstruction Quality Check ──────────────────────────────────────
    ae.eval()
    with torch.no_grad():
        pet, mri, _ = test_ds[0]
        recon, _, _ = ae(pet.unsqueeze(0).to(DEVICE))

    import numpy as np
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    slices = [40, 48, 60]  # three different axial slices

    for col, mid in enumerate(slices):
        axes[0, col].imshow(pet[0, :, :, mid].numpy(), cmap="hot")
        axes[0, col].set_title(f"Real PET (slice {mid})")
        axes[0, col].axis("off")
        axes[1, col].imshow(recon[0, 0, :, :, mid].cpu().numpy(), cmap="hot")
        axes[1, col].set_title(f"AE Recon (slice {mid})")
        axes[1, col].axis("off")

    plt.suptitle(f"AE Reconstruction after {len(ae_losses)} epochs", fontsize=13)
    plt.tight_layout()
    plt.savefig("/scratch/network/sz3962/taugennet/ae_recon.png", dpi=150)
    plt.show()

    recon_np = recon[0, 0].cpu().numpy()
    pet_np   = pet[0].numpy()
    print(f"Reconstruction MSE: {np.mean((recon_np - pet_np)**2):.4f}")
    print(f"Reconstruction MAE: {np.mean(np.abs(recon_np - pet_np)):.4f}")
