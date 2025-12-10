import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class ExonetDataset(Dataset):
    def __init__(
        self,
        df,
        base_dir,
        virtual_classname=None,
        transform=None,
        target_transform=None
    ):
        self.df = df
        self.transform = transform
        self.virtual_classname = virtual_classname
        self.target_transform = target_transform
        self.images_base_dir = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return an image and its label
        """
        _class = self.df.at[idx, "class"]
        _frame = str(self.df.at[idx, "frame"])
        _video = self.df.at[idx, "video"]
        _virtual_class = self.df.at[idx, self.virtual_classname] if self.virtual_classname else None
        
        image_path = self.images_base_dir / _class / f"['{_video}'] frame {_frame}.jpg"
        frame = Image.open(image_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        if self.target_transform:
            _class = self.target_transform([_class if not _virtual_class else _virtual_class])[0]

        return frame, _class