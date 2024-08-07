import argparse
import os
import numpy as np
import datetime
import time
import sys

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import Generator, Encoder, MultiDiscriminator, weights_init_normal
from datasets import ImageDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of CPU threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--img_width", type=int, default=128, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
    parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
    parser.add_argument("--lambda_kl", type=float, default=0.01, help="Kullback-Leibler loss weight")
    return parser.parse_args()

def setup_directories(dataset_name):
    os.makedirs(os.path.join("images", dataset_name), exist_ok=True)
    os.makedirs(os.path.join("saved_models", dataset_name), exist_ok=True)

def load_models(opt, input_shape, cuda):
    generator = Generator(opt.latent_dim, input_shape)
    encoder = Encoder(opt.latent_dim, input_shape)
    D_VAE = MultiDiscriminator(input_shape)
    D_LR = MultiDiscriminator(input_shape)

    if cuda:
        generator = generator.cuda()
        encoder = encoder.cuda()
        D_VAE = D_VAE.cuda()
        D_LR = D_LR.cuda()

    if opt.epoch != 0:
        try:
            generator.load_state_dict(torch.load(os.path.join("saved_models", opt.dataset_name, f"generator_{opt.epoch}.pth")))
            encoder.load_state_dict(torch.load(os.path.join("saved_models", opt.dataset_name, f"encoder_{opt.epoch}.pth")))
            D_VAE.load_state_dict(torch.load(os.path.join("saved_models", opt.dataset_name, f"D_VAE_{opt.epoch}.pth")))
            D_LR.load_state_dict(torch.load(os.path.join("saved_models", opt.dataset_name, f"D_LR_{opt.epoch}.pth")))
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            print("Training from scratch.")
            generator.apply(weights_init_normal)
            D_VAE.apply(weights_init_normal)
            D_LR.apply(weights_init_normal)
    else:
        generator.apply(weights_init_normal)
        D_VAE.apply(weights_init_normal)
        D_LR.apply(weights_init_normal)

    return generator, encoder, D_VAE, D_LR

def save_models(opt, epoch, generator, encoder, D_VAE, D_LR):
    torch.save(generator.state_dict(), os.path.join("saved_models", opt.dataset_name, f"generator_{epoch}.pth"))
    torch.save(encoder.state_dict(), os.path.join("saved_models", opt.dataset_name, f"encoder_{epoch}.pth"))
    torch.save(D_VAE.state_dict(), os.path.join("saved_models", opt.dataset_name, f"D_VAE_{epoch}.pth"))
    torch.save(D_LR.state_dict(), os.path.join("saved_models", opt.dataset_name, f"D_LR_{epoch}.pth"))

def sample_images(generator, val_dataloader, batches_done, opt, Tensor):
    """Saves a generated sample from the validation set"""
    generator.eval()
    imgs = next(iter(val_dataloader))
    img_samples = None
    for img_A, img_B in zip(imgs["A"], imgs["B"]):
        real_A = img_A.view(1, *img_A.shape).repeat(opt.latent_dim, 1, 1, 1)
        real_A = Variable(real_A.type(Tensor))
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (opt.latent_dim, opt.latent_dim))))
        fake_B = generator(real_A, sampled_z)
        fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        img_sample = torch.cat((img_A, fake_B), -1)
        img_sample = img_sample.view(1, *img_sample.shape)
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples, os.path.join("images", opt.dataset_name, f"{batches_done}.png"), nrow=8, normalize=True)
    generator.train()

def reparameterization(mu, logvar, Tensor):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), mu.size(1)))))
    z = sampled_z * std + mu
    return z

def main():
    opt = parse_args()
    setup_directories(opt.dataset_name)
    cuda = torch.cuda.is_available()
    input_shape = (opt.channels, opt.img_height, opt.img_width)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Loss functions
    mae_loss = torch.nn.L1Loss()
    if cuda:
        mae_loss.cuda()

    generator, encoder, D_VAE, D_LR = load_models(opt, input_shape, cuda)

    # Optimizers
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Dataloaders
    dataloader = DataLoader(
        ImageDataset(os.path.join("../../data", opt.dataset_name), input_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    val_dataloader = DataLoader(
        ImageDataset(os.path.join("../../data", opt.dataset_name), input_shape, mode="val"),
        batch_size=8,
        shuffle=True,
        num_workers=1,
    )

    valid = 1
    fake = 0
    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            optimizer_E.zero_grad()
            optimizer_G.zero_grad()

            # cVAE-GAN
            mu, logvar = encoder(real_B)
            encoded_z = reparameterization(mu, logvar, Tensor)
            fake_B = generator(real_A, encoded_z)
            loss_pixel = mae_loss(fake_B, real_B)
            loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
            loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

            # cLR-GAN
            sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), opt.latent_dim))))
            _fake_B = generator(real_A, sampled_z)
            loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)

            # Total Loss (Generator + Encoder)
            loss_GE = loss_VAE_GAN + loss_LR_GAN + opt.lambda_pixel * loss_pixel + opt.lambda_kl * loss_kl
            loss_GE.backward(retain_graph=True)
            optimizer_E.step()

            # Generator Only Loss
            _mu, _ = encoder(_fake_B)
            loss_latent = opt.lambda_latent * mae_loss(_mu, sampled_z)
            loss_latent.backward()
            optimizer_G.step()

            # Train Discriminator (cVAE-GAN)
            optimizer_D_VAE.zero_grad()
            loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(fake_B.detach(), fake)
            loss_D_VAE.backward()
            optimizer_D_VAE.step()

            # Train Discriminator (cLR-GAN)
            optimizer_D_LR.zero_grad()
            loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(_fake_B.detach(), fake)
            loss_D_LR.backward()
            optimizer_D_LR.step()

            # Log Progress
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                f"\r[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] [D VAE_loss: {loss_D_VAE.item():.6f}, LR_loss: {loss_D_LR.item():.6f}] [G loss: {loss_GE.item():.6f}, pixel: {loss_pixel.item():.6f}, kl: {loss_kl.item():.6f}, latent: {loss_latent.item():.6f}] ETA: {time_left}"
            )

            if batches_done % opt.sample_interval == 0:
                sample_images(generator, val_dataloader, batches_done, opt, Tensor)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            save_models(opt, epoch, generator, encoder, D_VAE, D_LR)

if __name__ == "__main__":
    main()
