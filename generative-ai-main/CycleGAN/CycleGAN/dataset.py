from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None, target_attr=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_attr = target_attr

        # Load annotations
        annotations_file = os.path.join(root_dir, "list_attr_celeba.txt")
        self.annotations = pd.read_csv(annotations_file, delim_whitespace=True, header=1)

        # Get the list of image file names
        self.image_names = self.annotations["image_id"].values

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        # Get the image file name
        image_name = self.image_names[index]
        image_path = os.path.join(self.root_dir, "img_align_celeba", image_name)

        # Load the image
        image = np.array(Image.open(image_path).convert("RGB"))

        # Get the target attribute value (if specified)
        if self.target_attr is not None:
            target = self.annotations.loc[index, self.target_attr]
        else:
            target = None

        # Apply transformations
        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, target