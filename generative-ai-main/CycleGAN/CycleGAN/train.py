import torch
from dataset import CelebADataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from Discriminator_model import Discriminator
from Generator_model import Generator

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (image, target) in enumerate(loop):
        image = image.to(config.DEVICE)
        target = target.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_image = gen_H(image)
            D_H_real = disc_H(image, target)
            D_H_fake = disc_H(fake_image.detach(), target)
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_image = gen_Z(image)
            D_Z_real = disc_Z(image, target)
            D_Z_fake = disc_Z(fake_image.detach(), target)
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            fake_image = gen_H(image)
            D_H_fake = disc_H(fake_image, target)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))

            fake_image = gen_Z(image)
            D_Z_fake = disc_Z(fake_image, target)
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            G_loss = loss_G_Z + loss_G_H

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_image * 0.5 + 0.5, f"saved_images/image_{idx}.png")

def main():
    disc_H = Discriminator(in_channels=3, target_dim=2).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3, target_dim=2).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE)

    # Dataset 1: Men without glasses and men with glasses
    dataset1 = CelebADataset(
        root_dir=config.TRAIN_DIR,
        transform=config.transforms,
        target_attr=["Male", "Glasses"],
        target_values=[[1, 0], [1, 1]]
    )

    # Dataset 2: Men with glasses and women with glasses
    dataset2 = CelebADataset(
        root_dir=config.TRAIN_DIR,
        transform=config.transforms,
        target_attr=["Male", "Glasses"],
        target_values=[[1, 1], [0, 1]]
    )

    val_dataset = CelebADataset(
        root_dir="path/to/val/dir",
        transform=config.transforms,
        target_attr=["Male", "Glasses"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    loader1 = DataLoader(
        dataset1,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    loader2 = DataLoader(
        dataset2,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader1, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader2, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

    # Testing
    with torch.no_grad():
        # Men without glasses to men with glasses (5 images)
        for i in range(5):
            image, target = next(iter(val_loader))
            image = image.to(config.DEVICE)
            target = target.to(config.DEVICE)
            if target[0, 0] == 1 and target[0, 1] == 0:  # Men without glasses
                fake_image = gen_H(image)
                save_image(fake_image * 0.5 + 0.5, f"test_results/men_without_to_with_{i}.png")

        # Men with glasses to men without glasses (5 images)
        for i in range(5):
            image, target = next(iter(val_loader))
            image = image.to(config.DEVICE)
            target = target.to(config.DEVICE)
            if target[0, 0] == 1 and target[0, 1] == 1:  # Men with glasses
                fake_image = gen_Z(image)
                save_image(fake_image * 0.5 + 0.5, f"test_results/men_with_to_without_{i}.png")
        # Men with glasses to women with glasses (5 images)
        for i in range(5):
            image, target = next(iter(val_loader))
            image = image.to(config.DEVICE)
            target = target.to(config.DEVICE)
            if target[0, 0] == 1 and target[0, 1] == 1:  # Men with glasses
                fake_image = gen_H(image)
                save_image(fake_image * 0.5 + 0.5, f"test_results/men_with_to_women_with_{i}.png")

        # Women with glasses to men with glasses (5 images)
        for i in range(5):
            image, target = next(iter(val_loader))
            image = image.to(config.DEVICE)
            target = target.to(config.DEVICE)
            if target[0, 0] == 0 and target[0, 1] == 1:  # Women with glasses
                fake_image = gen_Z(image)
                save_image(fake_image * 0.5 + 0.5, f"test_results/women_with_to_men_with_{i}.png")
    if __name__ == "__main__":
        main()