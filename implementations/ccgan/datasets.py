import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_x=None, transforms_lr=None, mode='train'):
        self.transform_x = transforms.Compose(transforms_x) if transforms_x else None
        self.transform_lr = transforms.Compose(transforms_lr) if transforms_lr else None

        # Filter to include only image files
        self.files = sorted(glob.glob(f'{root}/*.*'))
        self.files = [file for file in self.files if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]

    def __getitem__(self, index):
        try:
            img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        except Exception as e:
            print(f"Error loading image {self.files[index % len(self.files)]}: {e}")
            return None

        x = self.transform_x(img) if self.transform_x else img
        x_lr = self.transform_lr(img) if self.transform_lr else img

        return {'x': x, 'x_lr': x_lr}

    def __len__(self):
        return len(self.files)
