import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train"):
        self.transform = transforms.Compose([
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):
        while True:
            try:
                img = Image.open(self.files[index % len(self.files)]).convert('RGB')
                break
            except Exception as e:
                print(f"Error loading image {self.files[index % len(self.files)]}: {e}")
                index = (index + 1) % len(self.files)
        
        w, h = img.size
        img_A = img.crop((0, 0, w // 2, h))
        img_B = img.crop((w // 2, 0, w, h))

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):

        return len(self.files)
