import argparse
import os
import numpy as np
import itertools
import datetime
import time
import logging

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import torch

from models import *
from datasets import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def prepare_directories(dataset_name):
    os.makedirs(f"images/{dataset_name}", exist_ok=True)
    os.makedirs(f"saved_models/{dataset_name}", exist_ok=True)

def initialize_models(input_shape, n_residual_blocks, cuda):
    G_AB = GeneratorResNet(input_shape, n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)
    if cuda:
        G_AB.cuda()
        G_BA.cuda()
        D_A.cuda()
        D_B.cuda()
    return G_AB, G_BA, D_A, D_B

def load_pretrained_models(epoch, dataset_name, G_AB, G_BA, D_A, D_B):
    G_AB.load_state_dict(torch.load(f"saved_models/{dataset_name}/G_AB_{epoch}.pth"))
    G_BA.load_state_dict(torch.load(f"saved_models/{dataset_name}/G_BA_{epoch}.pth"))
    D_A.load_state_dict(torch.load(f"saved_models/{dataset_name}/D_A_{epoch}.pth"))
    D_B.load_state_dict(torch.load(f"saved_models/{dataset_name}/D_B_{epoch}.pth"))

def initialize_losses(cuda):
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    if cuda:
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
    return criterion_GAN, criterion_cycle, criterion_identity

def initialize_optimizers(G_AB, G_BA, D_A, D_B, lr, b1, b2):
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))
    return optimizer_G, optimizer_D_A, optimizer_D_B

def initialize_schedulers(optimizer_G, optimizer_D_A, optimizer_D_B, n_epochs, epoch, decay_epoch):
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
    return lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B

def sample_images(batches_done, val_dataloader, dataset_name, G_AB, G_BA, Tensor):
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, f"images/{dataset_name}/{batches_done}.png", normalize=False)

def train(opt):
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    input_shape = (opt.channels, opt.img_height, opt.img_width)

    G_AB, G_BA, D_A, D_B = initialize_models(input_shape, opt.n_residual_blocks, cuda)
    if opt.epoch != 0:
        load_pretrained_models(opt.epoch, opt.dataset_name, G_AB, G_BA, D_A, D_B)
    else:
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    criterion_GAN, criterion_cycle, criterion_identity = initialize_losses(cuda)
    optimizer_G, optimizer_D_A, optimizer_D_B = initialize_optimizers(G_AB, G_BA, D_A, D_B, opt.lr, opt.b1, opt.b2)
    lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B = initialize_schedulers(optimizer_G, optimizer_D_A, optimizer_D_B, opt.n_epochs, opt.epoch, opt.decay_epoch)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    transforms_ = [
        transforms.Resize(int(opt.img_height * 1.12), transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset(f"../../data/{opt.dataset_name}", transforms_=transforms_, unaligned=True),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    val_dataloader = DataLoader(
        ImageDataset(f"../../data/{opt.dataset_name}", transforms_=transforms_, unaligned=True, mode="test"),
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator A
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # Train Discriminator B
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # Log progress
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            logging.info(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}, adv: {loss_GAN.item()}, cycle: {loss_cycle.item()}, identity: {loss_identity.item()}] ETA: {time_left}")

            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done, val_dataloader, opt.dataset_name, G_AB, G_BA, Tensor)

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(G_AB.state_dict(), f"saved_models/{opt.dataset_name}/G_AB_{epoch}.pth")
            torch.save(G_BA.state_dict(), f"saved_models/{opt.dataset_name}/G_BA_{epoch}.pth")
            torch.save(D_A.state_dict(), f"saved_models/{opt.dataset_name}/D_A_{epoch}.pth")
            torch.save(D_B.state_dict(), f"saved_models/{opt.dataset_name}/D_B_{epoch}.pth")

if __name__ == "__main__":
    opt = parse_args()
    setup_logging()
    prepare_directories(opt.dataset_name)
    train(opt)
