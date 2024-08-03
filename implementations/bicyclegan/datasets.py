import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train"):
        """
        Initializes the dataset with the given root directory, image shape, and mode (train/test).
        Args:
            root (str): Root directory of the dataset.
            input_shape (tuple): Shape of the input images.
            mode (str): Mode for loading data, either 'train' or 'test'.
        """
        self.transform = transforms.Compose([
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):
        """
        Returns a single data pair from the dataset.
        Args:
            index (int): Index of the data to be fetched.
        Returns:
            dict: Dictionary containing the image pair 'A' and 'B'.
        """
        try:
            img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        except Exception as e:
            print(f"Error loading image {self.files[index % len(self.files)]}: {e}")
            return None

        w, h = img.size
        img_A = img.crop((0, 0, w // 2, h))
        img_B = img.crop((w // 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        """
        Returns the total number of files in the dataset.
        Returns:
            int: Number of files.
        """
        return len(self.files)
