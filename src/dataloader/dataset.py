import os
import typing as t

import numpy as np
import polars as pl
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset


class NYUDataset(Dataset):
    _data_path_ = "_db_"

    def __init__(self, params: t.Dict[str, any], stage: str = "train"):
        super().__init__()
        h, w = params["dataset"]["size"]
        self.stage = stage
        dataset_stage = "train" if stage in ["train", "valid"] else "test"
        self.anno_path = os.path.join(self._data_path_, "data", f"nyu2_{dataset_stage}.csv")
        self.anno = pl.read_csv(source=self.anno_path, has_header=False)
        self._split_data()
        # if not self.stage == "test":
        #     self._split_data()
        self.resize = A.Resize(h, w)
        self.transforms_img = A.Compose([
            A.RandomBrightnessContrast(),
            A.Normalize()
        ])
        self.dropout = A.CoarseDropout(
            min_holes=int(h*w*0.89),
            max_holes=int(h*w*0.91),
            max_width=1,
            max_height=1,
            p=1
        )
        self.toTensor = ToTensorV2()
    
    def _split_data(self):
        split_value = int(len(self.anno) * 0.9)
        if self.stage == "train":
            self.anno = self.anno[:split_value]
        elif self.stage == "valid":
            self.anno = self.anno[split_value:]

    def _transforms(self, img, depth):
        img = self.resize(image=img)["image"]
        img = self.transforms_img(image=img)["image"]
        img = self.toTensor(image=img)["image"]
        depth = self.resize(image=depth)["image"]
        additional_channel = self.toTensor(image=self.dropout(image=depth)["image"])["image"]
        img = torch.cat([img, additional_channel], dim=0)
        depth = self.toTensor(image=depth)["image"]
        return img / 255., depth / 255.

    def __getitem__(self, idx):
        img_path, depth_path = (os.path.join(self._data_path_, item) for item in self.anno.row(idx))
        img, depth = np.array(Image.open(img_path)), np.array(Image.open(depth_path))
        img, depth = self._transforms(img, depth)
        return img, depth

    def __len__(self):
        return len(self.anno)

if __name__ == "__main__":
    dataset = NYUDataset(stage="train")
    data = dataset[0]
    img, depth = data
    print(depth.min(), depth.max())

