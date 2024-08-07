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

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    zn = Variable(Tensor(0.75 * np.random.normal(0, 1, (shape, latent_dim))), requires_grad=req_grad)
    zc_FT = Tensor(shape, n_c).fill_(0)
    zc_idx = torch.empty(shape, dtype=torch.long)

    if fix_class == -1:
        zc_idx = zc_idx.random_(n_c).cuda() if torch.cuda.is_available() else zc_idx.random_(n_c)
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

        zc_idx = zc_idx.cuda() if torch.cuda.is_available() else zc_idx
        zc_FT = zc_FT.cuda() if torch.cuda.is_available() else zc_FT

    zc = Variable(zc_FT, requires_grad=req_grad)

    return zn, zc, zc_idx

def calc_gradient_penalty(netD, real_data, generated_data, LAMBDA=10):
    b_size = real_data.size()[0]
    alpha = torch.rand(b_size, 1, 1, 1).expand_as(real_data).cuda() if torch.cuda.is_available() else torch.rand(b_size, 1, 1, 1).expand_as(real_data)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True).cuda() if torch.cuda.is_available() else Variable(interpolated, requires_grad=True)

    prob_interpolated = netD(interpolated)
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda() if torch.cuda.is_available() else torch.ones(prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(b_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    return LAMBDA * ((gradients_norm - 1) ** 2).mean()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

def softmax(x):
    return F.softmax(x, dim=1)

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
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
        )

        initialize_weights(self)

        if self.verbose:
            print(f"Setting up {self.name}...\n")
            print(self.model)
    
    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model(z)
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen

class Encoder_CNN(nn.Module):
    def __init__(self, latent_dim, n_c, verbose=False):
        super(Encoder_CNN, self).__init__()
        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose
        
        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(self.lshape),
            nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, latent_dim + n_c)
        )

        initialize_weights(self)
        
        if self.verbose:
            print(f"Setting up {self.name}...\n")
            print(self.model)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        z = z_img.view(z_img.shape[0], -1)
        zn = z[:, :self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        zc = softmax(zc_logits)
        return zn, zc, zc_logits

class Discriminator_CNN(nn.Module):
    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()
        self.name = 'discriminator'
        self.channels = 1
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        self.verbose = verbose
        
        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(self.lshape),
            nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )
        
        if not self.wass:
            self.model.add_module('sigmoid', nn.Sigmoid())

        initialize_weights(self)

        if self.verbose:
            print(f"Setting up {self.name}...\n")
            print(self.model)

    def forward(self, img):
        validity = self.model(img)
        return validity

def main():
    create_directories()
    args = parse_arguments()

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    test_batch_size = 5000
    lr = args.learning_rate
    b1 = 0.5
    b2 = 0.9
    decay = 2.5e-5
    n_skip_iter = args.n_critic

    # Data dimensions
    img_size = args.img_size
    channels = 1

    # Latent space info
    latent_dim = args.latent_dim
    n_c = 10
    betan = 10
    betac = 10

    # Wasserstein+GP metric flag
    wass_metric = args.wass_flag

    x_shape = (channels, img_size, img_size)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    # Loss functions
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    # Initialize models
    generator = Generator_CNN(latent_dim, n_c, x_shape, verbose=True)
    encoder = Encoder_CNN(latent_dim, n_c, verbose=True)
    discriminator = Discriminator_CNN(wass_metric=wass_metric, verbose=True)

    if cuda:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    testdata = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_imgs, test_labels = next(iter(testdata))
    test_imgs = Variable(test_imgs.type(Tensor))

    ge_chain = ichain(generator.parameters(), encoder.parameters())

    optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Training loop
    ge_l = []
    d_l = []

    c_zn = []
    c_zc = []
    c_i = []

    print(f'\nBegin training session with {n_epochs} epochs...\n')
    for epoch in range(n_epochs):
        for i, (imgs, itruth_label) in enumerate(dataloader):
            generator.train()
            encoder.train()
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()

            real_imgs = Variable(imgs.type(Tensor))

            optimizer_GE.zero_grad()

            zn, zc, zc_idx = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=n_c)

            gen_imgs = generator(zn, zc)

            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)

            if i % n_skip_iter == 0:
                enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)
                zn_loss = mse_loss(enc_gen_zn, zn)
                zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)

                if wass_metric:
                    ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss
                else:
                    valid = Variable(Tensor(gen_imgs.size(0), 1).fill_(1.0), requires_grad=False)
                    v_loss = bce_loss(D_gen, valid)
                    ge_loss = v_loss + betan * zn_loss + betac * zc_loss

                ge_loss.backward(retain_graph=True)
                optimizer_GE.step()

            optimizer_D.zero_grad()

            if wass_metric:
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)
                d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
            else:
                fake = Variable(Tensor(gen_imgs.size(0), 1).fill_(0.0), requires_grad=False)
                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        d_l.append(d_loss.item())
        ge_l.append(ge_loss.item())

        generator.eval()
        encoder.eval()

        n_samp = 25

        t_imgs, t_label = test_imgs.data, test_labels
        e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
        teg_imgs = generator(e_tzn, e_tzc)
        img_mse_loss = mse_loss(t_imgs, teg_imgs)
        c_i.append(img_mse_loss.item())

        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp, latent_dim=latent_dim, n_c=n_c)
        gen_imgs_samp = generator(zn_samp, zc_samp)
        zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)
        lat_mse_loss = mse_loss(zn_e, zn_samp)
        lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)
        c_zn.append(lat_mse_loss.item())
        c_zc.append(lat_xe_loss.item())

        r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        e_zn, e_zc, e_zc_logits = encoder(r_imgs)
        reg_imgs = generator(e_zn, e_zc)
        save_image(reg_imgs.data[:n_samp], f'images/cycle_reg_{epoch:06d}.png', nrow=5, normalize=True)
        save_image(gen_imgs_samp.data[:n_samp], f'images/gen_{epoch:06d}.png', nrow=5, normalize=True)

        stack_imgs = []
        for idx in range(n_c):
            zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_c, latent_dim=latent_dim, n_c=n_c, fix_class=idx)
            gen_imgs_samp = generator(zn_samp, zc_samp)
            stack_imgs = gen_imgs_samp if len(stack_imgs) == 0 else torch.cat((stack_imgs, gen_imgs_samp), 0)

        save_image(stack_imgs, f'images/gen_classes_{epoch:06d}.png', nrow=n_c, normalize=True)

        print(f"[Epoch {epoch}/{n_epochs}] \n\tModel Losses: [D: {d_loss.item():.6f}] [GE: {ge_loss.item():.6f}]")
        print(f"\tCycle Losses: [x: {img_mse_loss.item():.6f}] [z_n: {lat_mse_loss.item():.6f}] [z_c: {lat_xe_loss.item():.6f}]")

if __name__ == "__main__":
    main()
