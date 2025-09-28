# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from models import Generator, Discriminator, NZ
from data import get_mnist_dataloader
from utils_gan import ensure_dir, save_grid, show_progress, sample_fixed

# -------------------
# Hyperparameters
# -------------------
BATCH_SIZE = 128
IMG_SIZE = 28
LATENT_DIM = NZ          # keep in sync with models.py
EPOCHS = 30
LR = 2e-4
BETA1, BETA2 = 0.5, 0.999
OUT_DIR = "outputs"      # where images go

# -------------------
# Setup
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ensure_dir(OUT_DIR)

netG = Generator(nz=LATENT_DIM).to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))

dataloader = get_mnist_dataloader(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
real_label, fake_label = 1.0, 0.0

saved_grids = []

# -------------------
# Training Loop
# -------------------
for epoch in range(EPOCHS):
    for i, (real, _) in enumerate(dataloader):
        real = real.to(device)
        bsz = real.size(0)

        # ---- Train Discriminator ----
        netD.zero_grad()
        # real
        out_real = netD(real)
        label_real = torch.full_like(out_real, real_label, device=device)
        loss_real = criterion(out_real, label_real)
        loss_real.backward()

        # fake
        noise = torch.randn(bsz, LATENT_DIM, 1, 1, device=device)
        fake = netG(noise)
        out_fake = netD(fake.detach())
        label_fake = torch.full_like(out_fake, fake_label, device=device)
        loss_fake = criterion(out_fake, label_fake)
        loss_fake.backward()
        optimizerD.step()

        # ---- Train Generator ----
        netG.zero_grad()
        out_fake_forG = netD(fake)
        label_forG = torch.full_like(out_fake_forG, real_label, device=device)
        loss_G = criterion(out_fake_forG, label_forG)
        loss_G.backward()
        optimizerG.step()

    # ---- Epoch logging ----
    loss_D = (loss_real + loss_fake).item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss_D: {loss_D:.4f} | Loss_G: {loss_G.item():.4f}")

    # ---- Save samples from fixed noise ----
    with torch.no_grad():
        samples = sample_fixed(netG, fixed_noise)
        grid = vutils.make_grid(samples, padding=2, normalize=True)
        saved_grids.append(grid)
        fp = os.path.join(OUT_DIR, f"fake_epoch_{epoch+1:03d}.png")
        vutils.save_image(grid, fp)

# -------------------
# Optional: show montage in one figure (interactive usage)
# -------------------
try:
    show_progress(saved_grids, rows=6, cols=5, title_prefix="Epoch")
except Exception as e:
    # headless envs may not have display; it's fine if this fails
    pass
