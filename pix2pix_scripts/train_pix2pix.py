import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from dataset import Pix2PixDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Config ===
image_size = 256
batch_size = 4
lr = 2e-4
epochs = 100
save_every = 100
start_epoch = 0  # Change this to resume from the last saved epoch.

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Dataset & DataLoader ===
dataset = Pix2PixDataset(
    root_sketch='dataset/raw_sketches/dog',
    root_real='dataset/refined_sketches/dog',
    transform=transform
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Models ===
gen = UNetGenerator().to(device)
disc = PatchDiscriminator().to(device)

# === Optimizers ===
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# === Load Checkpoints ===
# The default resume path is `checkpoints2/`; published checkpoints can be
# downloaded or linked there before resuming training.
gen_ckpt = f"checkpoints2/gen_epoch{start_epoch}.pth"
disc_ckpt = f"checkpoints2/disc_epoch{start_epoch}.pth"

if os.path.exists(gen_ckpt) and os.path.exists(disc_ckpt):
    gen.load_state_dict(torch.load(gen_ckpt, map_location=device))
    disc.load_state_dict(torch.load(disc_ckpt, map_location=device))
    print(f"Resumed from epoch {start_epoch}")
else:
    print(f"Checkpoints not found for epoch {start_epoch}, starting from scratch")
    start_epoch = 0

# === Losses ===
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

# === Make output dirs ===
os.makedirs("outputs2", exist_ok=True)
os.makedirs("checkpoints2", exist_ok=True)

# === Training Loop ===
for epoch in range(start_epoch + 1, epochs + 1):
    loop = tqdm(loader, desc=f"Epoch [{epoch}/{epochs}]", leave=True)
    for idx, (sketch, real) in enumerate(loop):
        sketch, real = sketch.to(device), real.to(device)

        # === Generate fake image ===
        fake = gen(sketch)

        # === Discriminator ===
        disc_real = disc(sketch, real)
        disc_fake = disc(sketch, fake.detach())

        loss_disc_real = criterion_GAN(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = criterion_GAN(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # === Generator ===
        disc_fake_for_gen = disc(sketch, fake)
        loss_gen_gan = criterion_GAN(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
        loss_gen_l1 = criterion_L1(fake, real) * 100
        loss_gen = loss_gen_gan + loss_gen_l1

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        loop.set_postfix(loss_gen=loss_gen.item(), loss_disc=loss_disc.item())

        # Save sample
        if idx % save_every == 0:
            save_image(fake * 0.5 + 0.5, f"outputs2/fake_{epoch}_{idx}.png")

    # Save checkpoint after every epoch
    torch.save(gen.state_dict(), f"checkpoints2/gen_epoch{epoch}.pth")
    torch.save(disc.state_dict(), f"checkpoints2/disc_epoch{epoch}.pth")
