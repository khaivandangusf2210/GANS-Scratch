import argparse
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def create_directories():
    os.makedirs("images", exist_ok=True)
    os.makedirs("../../data/mnist", exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    return parser.parse_args()

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

def boundary_seeking_loss(y_pred, y_true):
    """
    Boundary seeking loss.
    Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
    """
    return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2)

def load_data(img_size, batch_size):
    dataloader = DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader

def train(generator, discriminator, dataloader, opt, device):
    discriminator_loss = nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            valid = Variable(torch.ones(batch_size, 1).to(device), requires_grad=False)
            fake = Variable(torch.zeros(batch_size, 1).to(device), requires_grad=False)

            real_imgs = Variable(imgs.type(torch.FloatTensor).to(device))

            # Train Generator
            optimizer_G.zero_grad()

            z = Variable(torch.randn(batch_size, opt.latent_dim).to(device))
            gen_imgs = generator(z)

            g_loss = boundary_seeking_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            validity_real = discriminator(real_imgs)
            validity_fake = discriminator(gen_imgs.detach())
            d_real_loss = discriminator_loss(validity_real, valid)
            d_fake_loss = discriminator_loss(validity_fake, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]"
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], f"images/{batches_done}.png", nrow=5, normalize=True)

def main():
    create_directories()
    opt = parse_arguments()
    img_shape = (opt.channels, opt.img_size, opt.img_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(opt.latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)

    dataloader = load_data(opt.img_size, opt.batch_size)

    train(generator, discriminator, dataloader, opt, device)

if __name__ == "__main__":
    main()
