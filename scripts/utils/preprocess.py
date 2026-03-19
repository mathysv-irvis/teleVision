import torch
import os

import pandas as pd
from PIL import Image

class ArtifactDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(row["fname"], "art_cam_blur.png")
        image = Image.open(img_path).convert("RGB")

        label = torch.tensor([
            row["pixel_art"],
            row["color_art"],
            row["column_art"]
        ], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

