import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_x=None, transforms_lr=None):
        self.transform_x = transforms.Compose(transforms_x) if transforms_x else lambda x: x
        self.transform_lr = transforms.Compose(transforms_lr) if transforms_lr else lambda x: x
        
        # Collecting valid image files
        self.files = sorted([f for f in glob.glob(os.path.join(root, '*')) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))])

    def __getitem__(self, index):
        file_path = self.files[index % len(self.files)]
        try:
            img = Image.open(file_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {file_path}: {e}")

        return {
            'x': self.transform_x(img),
            'x_lr': self.transform_lr(img)
        }

    def __len__(self):
        return len(self.files)
