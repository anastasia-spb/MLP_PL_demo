import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def is_image(filename: str, folder_path: str):
    if os.path.isfile(os.path.join(folder_path, filename)):
        _, file_extension = os.path.splitext(filename)
        return file_extension.lower() in {".jpg", ".png", ".jpeg"}
    return False


class MNISTEvalDataset(Dataset):
    def __init__(self, img_folder: str, transform=None):
        super().__init__()
        self.img_folder = img_folder
        self.images_names = [f for f in os.listdir(self.img_folder) if is_image(f, self.img_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index):
        img_name = self.images_names[index]
        gt, _ = os.path.splitext(img_name)
        gt = int(gt)
        with Image.open(os.path.join(self.img_folder, img_name)) as image:
            if self.transform is not None:
                image = self.transform(image)
        return image, gt
