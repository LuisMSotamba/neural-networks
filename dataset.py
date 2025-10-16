import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class ExoNetDatasetV2(Dataset):
    def __init__(
        self,
        df_labels: pd.DataFrame,
        seq_len: int,
        transform=None,
        target_transform=None
    ):
        self.df_data = df_labels
        self.seq_len = seq_len
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        """
        Return an image and its label
        """
        sequence = self.df_data.at[idx, "sequence"]
        image_paths = sequence.split(",")
        
        frames = [
            Image.open(
                path
            ).convert("RGB") for path in image_paths
        ]
        label = self.df_data.at[idx, "class"]
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        if self.target_transform:
            label = self.target_transform([label])

        frames_tensor = torch.stack(frames)
        label_tensor = torch.from_numpy(label)

        return frames_tensor, label_tensor

class ExoNetDataset(Dataset):
    def __init__(
        self,
        df_labels: pd.DataFrame,
        seq_len: int,
        img_dir: str,
        transform=None,
        target_transform=None
    ):
        self.img_labels = df_labels
        self.seq_len = seq_len
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)//self.seq_len

    def __getitem__(self, idx):
        """
        Return an image and its label
        """
        start_seq_idx = idx * self.seq_len
        end_seq_idx = (idx+1) * self.seq_len
        file_name_format = "['{}'] frame {}.jpg"
        frames = [
            Image.open(
                os.path.join(
                    self.img_dir,
                    self.img_labels.iloc[i, 2],
                    file_name_format.format(
                        self.img_labels.iloc[i, 0],
                        self.img_labels.iloc[i, 1]
                    )
                )
            ).convert("RGB") for i in range(start_seq_idx, end_seq_idx)
        ]
        labels = [self.img_labels.iloc[i, 2] for i in range(start_seq_idx, end_seq_idx)]        
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        if self.target_transform:
            labels = self.target_transform(labels)

        frames_tensor = torch.stack(frames)
        labels_tensor = torch.from_numpy(labels)

        return frames_tensor, labels_tensor