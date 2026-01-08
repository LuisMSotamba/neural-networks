import os
import torch
import pandas as pd
import numpy as np

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class VideoDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataframe: pd.DataFrame,
            img_dir: str,
            transform,
        ):
        super().__init__()
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_name = f"['{row['video']}'] frame {row['frame']}.jpg"
        image_path = os.path.join(self.img_dir, row["class"] ,image_name)
        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)
        
        return img

class ExonetDataset(Dataset):
    def __init__(
        self,
        df,
        base_dir,
        virtual_classname=None,
        transform=None,
        target_transform=None,
        return_filepath=False
    ):
        self.df = df
        self.transform = transform
        self.virtual_classname = virtual_classname
        self.target_transform = target_transform
        self.images_base_dir = base_dir
        self.return_filepath = return_filepath

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

        filename = f"{_class}/['{_video}'] frame {_frame}.jpg"
        image_path = self.images_base_dir / filename
        frame = Image.open(image_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        if self.target_transform:
            _class = self.target_transform([_class if not _virtual_class else _virtual_class])[0]

        if self.return_filepath:
            return frame, _class, filename
        return frame, _class

class FeSequenceExonetDataset(Dataset):
    """
    Loads feature sequences of each Exonet's frame
    """
    def __init__(
        self,
        df,
        seq_len,
        target_transform
    ):
        self.df = df
        self.target_transform = target_transform
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df) - self.seq_len + 1

    def __getitem__(self, idx):
        """
        Return a feature representation of a frame and its label
        """
        img_paths = self.df.at[idx, "sequence"].split(",")
        _class = self.df.at[idx, "class"]

        # read all sequence's frames
        features = [ np.load(path) for path in img_paths ]
        features_tensor = torch.tensor(np.stack(features), dtype=torch.float32)
        
        # transform class according to the defined target transform
        if self.target_transform:
            _class = self.target_transform([_class])[0]

        return features_tensor, _class

class FeExonetDataset(Dataset):
    """
    Load embeddings from a dataset
    """
    def __init__(
        self,
        df,
        target_transform,
        img_dir
    ):
        self.df = df
        self.target_transform = target_transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return an embedding
        """
        _class = self.df.at[idx, "class"]
        _frame = str(self.df.at[idx, "frame"])
        _video = self.df.at[idx, "video"]

        embedding_name = f"{_class}/['{_video}'] frame {_frame}.npy"
        image_name = f"{_class}/['{_video}'] frame {_frame}.jpg"
        
        embedding_path = self.img_dir / embedding_name
        embedding = np.load(embedding_path)

        if self.target_transform:
            _class = self.target_transform([_class])[0]
            
        return embedding, _class, image_name
        