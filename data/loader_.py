import os
import torch as t

from torch.utils.data import Dataset

class Vox256Embedding(Dataset):
    
    def __init__(self, ds_path: str):
        self.ds_path = ds_path
        self.videos = os.listdir(self.ds_path)
        
    def __getitem__(self, idx: int) -> t.Tensor:
        return t.load(self.ds_path + "/" + self.videos[idx])
    
    def __len__(self):
        return len(self.videos)
