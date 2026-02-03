import os
import torch
import pandas as pd
import numpy as np

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class WalkingLSTMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        seq_len: int,
        df: pd.DataFrame,
        features_dir: str,
        target_transform
    ):
        self.seq_len = seq_len
        self.df = df
        self.features_dir = features_dir
        self.target_transform = target_transform

        # Create a list of sequences
        self.samples = []

        for video_name, group in self.df.groupby('video'):
            # IMPORTANT: Sort by frame to ensure temporal order
            group = group.sort_values('frame', ascending=True).reset_index()
            
            # If video is shorter than sequence length, skip it
            if len(group) < self.seq_len:
                continue
            
            # Create Sliding Windows
            # Range logic: Stops so the last window fits exactly at the end
            for i in range(0, len(group) - self.seq_len + 1, 1):
                
                # The label is usually the label of the LAST frame in the sequence
                # (We are predicting the current state based on history)
                idx = group.iloc[i]['index']
                
                # Store the pointer (Video Name, Start Index)
                self.samples.append((video_name, idx))
                
        print(f"Dataset Loaded: {len(self.samples)} sequences found across {df['video'].nunique()} videos.")

    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Retrieve pointer
        video_name, start_frame_idx = self.samples[idx]

        embedding_paths = [ os.path.join(
            self.features_dir, row["path"]
        ) for _, row in self.df.iloc[start_frame_idx: start_frame_idx + self.seq_len, :].iterrows()]

        target_class = self.df.at[start_frame_idx + self.seq_len -1, "class"]

        # Load embeddings
        embeddings = np.array([ np.load(path.replace(".jpg", ".npy")) for path in embedding_paths ])
        
        # Shape: (Sequence_Length, Feature_Dim) -> (16, 2048)
        X = torch.from_numpy(embeddings).float()
        y = self.target_transform([target_class])[0]
        
        return X, y

class WalkingDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataframe: pd.DataFrame,
            img_dir: str,
            transform,
            target_transform,
            return_filename=False
        ):
        super().__init__()
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.return_filename = return_filename

    
    def __len__(self):
        return len(self.dataframe)
    

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = os.path.join(self.img_dir, row["path"])
        _class = row["class"]
        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            _class = self.target_transform([_class])[0]

        if self.return_filename:
            return img, _class, row["path"]
        
        return img, _class


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
        