import os
import glob

import torch
import torchvision
from torch.utils.data import Dataset


class NYUDataset(Dataset):
    def __init__(self, stage):
        super().__init__()
        raise NotImplementedError("dataset.__init__")

    def __len__(self):
        raise NotImplementedError("dataset.__len__")

    def __getitem__(self, idx):
        raise NotImplementedError("dataset.__getitem__")


if __name__ == "__main__":
    dataset = NYUDataset(stage="train")
    data = dataset[0]


