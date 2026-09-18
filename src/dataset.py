"""
dataset.py
----------
Loads images from a folder and prepares them for the model:
  - resize to a fixed size (128x128)
  - convert to tensor
  - scale pixel values to [0, 1]

We use torchvision's ImageFolder-style loading but simplified since we
sometimes just have a flat folder of images (no class subfolders needed
for the 'good'-only training set).
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SIZE = 128

# Same transform is reused everywhere so train/test are treated identically.
default_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # converts PIL image [0-255] -> tensor [0-1]
])


class FolderImageDataset(Dataset):
    """Loads every image inside `folder_path` (non-recursive)."""

    def __init__(self, folder_path, transform=default_transform):
        self.folder_path = folder_path
        self.transform = transform
        valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(valid_ext)
        ]
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {folder_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, path
