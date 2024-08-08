from __future__ import print_function

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad as torch_grad
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image

from itertools import chain as ichain

def create_directories():
    os.makedirs("images", exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="ClusterGAN Training Script")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("-i", "--img_size", dest="img_size", type=int, default=28, help="Size of image dimension")
    parser.add_argument("-d", "--latent_dim", dest="latent_dim", default=30, type=int, help="Dimension of latent space")
    parser.add_argument("-l", "--lr", dest="learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-c", "--n_critic", dest="n_critic", type=int, default=5, help="Number of training steps for discriminator per iter")
    parser.add_argument("-w", "--wass_flag", dest="wass_flag", action='store_true', help="Flag for Wasserstein metric")
    return parser.parse_args()

def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False):
    assert (fix_class == -1 or (fix_class >= 0 and fix_class < n_c)), f"Requested class {fix_class} outside bounds."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    zn = torch.randn(shape, latent_dim, device=device, requires_grad=req_grad) * 0.75
    zc = torch.zeros(shape, n_c, device=device)
    zc_idx = torch.randint(n_c, (shape,), device=device) if fix_class == -1 else torch.full((shape,), fix_class, device=device)
    zc.scatter_(1, zc_idx.unsqueeze(1), 1.)
    
    return zn, zc, zc_idx

def calc_gradient_penalty(netD, real_data, generated_data, LAMBDA=10):
    device = real_data.device
    b_size = real_data.size(0)
    alpha = torch.rand(b_size, 1, 1, 1, device=device).expand_as(real_data)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated.requires_grad_(True)
    
    prob_interpolated = netD(interpolated)
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones_like(prob_interpolated),
                           create_graph=True, retain_graph=True)[0]
    
    gradients = gradients.view(b_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Reshape(nn.Module):
    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        return f'shape={self.shape}'

class Generator_CNN(nn.Module):
    def __init__(self, latent_dim, n_c, x_shape, verbose=False):
        super(Generator_CNN, self).__init__()
        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = (128, 7, 7)
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose
        
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim + self.n_c, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(self.ishape),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

        initialize_weights(self)

        if self.verbose:
            print(f"Setting up {self.name}...\n")
            print(self.model)
    
    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        return self.model(z).view(z.size(0), *self.x_shape)

class Encoder_CNN(nn.Module):
    def __init__(self, latent_dim, n_c, verbose=False):
        super(Encoder_CNN, self).__init__()
        self.name = 'encoder'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.verbose = verbose
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape((self.iels,)),
            nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, latent_dim + n_c)
        )

        initialize_weights(self)
        
        if self.verbose:
            print(f"Setting up {self.name}...\n")
            print(self.model)

    def forward(self, in_feat):
        z = self.model(in_feat).view(in_feat.size(0), -1)
        zn = z[:, :self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        zc = F.softmax(zc_logits, dim=1)
        return zn, zc, zc_logits

class Discriminator_CNN(nn.Module):
    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()
        self.name = 'discriminator'
        self.iels = int(np.prod((128, 5, 5)))
        self.wass = wass_metric
        self.verbose = verbose
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape((self.iels,)),
            nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            *(nn.Sigmoid(),) if not self.wass else ()
        )

        initialize_weights(self)

        if self.verbose:
            print(f"Setting up {self.name}...\n")
            print(self.model)

    def forward(self, img):
        return self.model(img)

def main():
    create_directories()
    args = parse_arguments()

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    n_skip_iter = args.n_critic

    # Data dimensions
    img_size = args.img_size
    x_shape = (1, img_size, img_size)

    # Latent space info
    latent_dim = args.latent_dim
    n_c = 10
    betan = 10
    betac = 10

    # Wasserstein+GP metric flag
    wass_metric = args.wass_flag

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss functions
    bce_loss = nn.BCELoss().to(device)
    xe_loss = nn.CrossEntropyLoss().to(device)
    mse_loss = nn.MSELoss().to(device)

    # Initialize models
    generator = Generator_CNN(latent_dim, n_c, x_shape, verbose=True).to(device)
    encoder = Encoder_CNN(latent_dim, n_c, verbose=True).to(device)
    discriminator = Discriminator_CNN(wass_metric=wass_metric, verbose=True).to(device)

    # Configure data loader
    dataloader = DataLoader(
        datasets.MNIST(
            "../../data/mnist", train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size, shuffle=True,
    )

    testdata = DataLoader(
        datasets.MNIST(
            "../../data/mnist", train=False, download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size, shuffle=True,
    )
    test_imgs, test_labels = next(iter(testdata))
    test_imgs = test_imgs.to(device)

    ge_chain = ichain(generator.parameters(), encoder.parameters())
    optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(0.5, 0.9), weight_decay=2.5e-5)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    # Training loop
    print(f'\nBegin training session with {n_epochs} epochs...\n')
    for epoch in range(n_epochs):
        for i, (imgs, itruth_label) in enumerate(dataloader):
            real_imgs = imgs.to(device)

            # -----------------
            #  Train Generator and Encoder
            # -----------------
            optimizer_GE.zero_grad()
            zn, zc, zc_idx = sample_z(shape=real_imgs.size(0), latent_dim=latent_dim, n_c=n_c)
            gen_imgs = generator(zn, zc)

            D_gen = discriminator(gen_imgs)

            if i % n_skip_iter == 0:
                enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)
                zn_loss = mse_loss(enc_gen_zn, zn)
                zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)

                if wass_metric:
                    ge_loss = D_gen.mean() + betan * zn_loss + betac * zc_loss
                else:
                    ge_loss = bce_loss(D_gen, torch.ones_like(D_gen)) + betan * zn_loss + betac * zc_loss

                ge_loss.backward(retain_graph=True)
                optimizer_GE.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            D_real = discriminator(real_imgs)

            if wass_metric:
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)
                d_loss = D_real.mean() - D_gen.mean() + grad_penalty
            else:
                d_loss = 0.5 * (bce_loss(D_real, torch.ones_like(D_real)) + bce_loss(D_gen, torch.zeros_like(D_gen)))

            d_loss.backward()
            optimizer_D.step()

        # Save generated samples at intervals
        if epoch % 10 == 0:
            generator.eval()
            encoder.eval()

            with torch.no_grad():
                reg_imgs = generator(encoder(real_imgs)[0], encoder(real_imgs)[1])
                save_image(reg_imgs[:25], f'images/cycle_reg_{epoch:06d}.png', nrow=5, normalize=True)

                stack_imgs = []
                for idx in range(n_c):
                    zn_samp, zc_samp, _ = sample_z(shape=n_c, latent_dim=latent_dim, n_c=n_c, fix_class=idx)
                    stack_imgs.append(generator(zn_samp, zc_samp))
                save_image(torch.cat(stack_imgs), f'images/gen_classes_{epoch:06d}.png', nrow=n_c, normalize=True)

            print(f"[Epoch {epoch}/{n_epochs}] D Loss: {d_loss.item():.6f}, GE Loss: {ge_loss.item():.6f}")

if __name__ == "__main__":
    main()
