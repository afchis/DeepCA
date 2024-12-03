import os
import glob

import polars as pl

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms as T


class NYUDataset(Dataset):
    _data_path_ = "_db_"

    def __init__(self, stage: str = "train"):
        super().__init__()
        self.stage = stage
        dataset_stage = "train" if stage in ["train", "valid"] else "test"
        self.anno_path = os.path.join(self._data_path, "data", f"nyu2_{dataset_stage}.scv")
        self.anno = pl.read_scv(self.anno_path)
        print(self.ano
        raise NotImplementedError("dataset.__init__")

    def __len__(self):
        raise NotImplementedError("dataset.__len__")

    def __getitem__(self, idx):
        raise NotImplementedError("dataset.__getitem__")


if __name__ == "__main__":
    dataset = NYUDataset(stage="train")
    data = dataset[0]


