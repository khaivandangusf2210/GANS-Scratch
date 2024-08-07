import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import argparse

from datasets import ImageDataset
from models import Generator, Discriminator, weights_init_normal

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of CPU threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    return parser.parse_args()

def setup_transforms(img_size):
    return [
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

def apply_random_mask(imgs, img_size, mask_size):
    masked_imgs = imgs.clone()
    for img in masked_imgs:
        x = np.random.randint(0, img_size - mask_size)
        y = np.random.randint(0, img_size - mask_size)
        img[:, y:y+mask_size, x:x+mask_size] = 0
    return masked_imgs

def save_samples_if_needed(saved_samples, real_imgs, masked_imgs, imgs_lr, batch_index, sample_interval):
    if batch_index % sample_interval == 0:
        save_image(real_imgs.data[:25], f"images/real_{batch_index}.png", nrow=5, normalize=True)
        save_image(masked_imgs.data[:25], f"images/masked_{batch_index}.png", nrow=5, normalize=True)
        save_image(imgs_lr.data[:25], f"images/lr_{batch_index}.png", nrow=5, normalize=True)

def train_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, cuda, Tensor, adversarial_loss, saved_samples, opt):
    for i, batch in enumerate(dataloader):
        imgs = batch["x"]
        imgs_lr = batch["x_lr"]
        masked_imgs = apply_random_mask(imgs, opt.img_size, opt.mask_size)

        valid = Tensor(imgs.size(0), *discriminator.output_shape).fill_(1.0)
        fake = Tensor(imgs.size(0), *discriminator.output_shape).fill_(0.0)

        real_imgs, imgs_lr, masked_imgs = map(lambda x: x.type(Tensor), [imgs, imgs_lr, masked_imgs])

        # Train Generator
        optimizer_G.zero_grad()
        gen_imgs = generator(masked_imgs, imgs_lr)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        print(f"[Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")
        save_samples_if_needed(saved_samples, real_imgs, masked_imgs, imgs_lr, i, opt.sample_interval)

def main():
    opt = setup_args()
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    os.makedirs("images", exist_ok=True)
    input_shape = (opt.channels, opt.img_size, opt.img_size)
    generator = Generator(input_shape).apply(weights_init_normal).cuda() if cuda else Generator(input_shape).apply(weights_init_normal)
    discriminator = Discriminator(input_shape).apply(weights_init_normal).cuda() if cuda else Discriminator(input_shape).apply(weights_init_normal)
    adversarial_loss = nn.MSELoss().cuda() if cuda else nn.MSELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    transforms_ = setup_transforms(opt.img_size)
    dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    saved_samples = {}
    for epoch in range(opt.n_epochs):
        print(f"Epoch {epoch+1}/{opt.n_epochs}")
        train_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, cuda, Tensor, adversarial_loss, saved_samples, opt)

if __name__ == "__main__":
    main()
