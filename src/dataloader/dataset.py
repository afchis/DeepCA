import os

import polars as pl
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset


class NYUDataset(Dataset):
    _data_path_ = "_db_"

    def __init__(self, stage: str = "train"):
        super().__init__()
        self.stage = stage
        dataset_stage = "train" if stage in ["train", "valid"] else "test"
        self.anno_path = os.path.join(self._data_path_, "data", f"nyu2_{dataset_stage}.csv")
        self.anno = pl.read_csv(source=self.anno_path, has_header=False)
        # if not self.stage == "test":
        #     self._split_data()
        self.transorms_img = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.RandomBrightnessContrast(),
                A.VerticalFlip(),
                ToTensorV2()
            ]
        )
        self.transorms_depth = A.Compose(
            [
                A.Resize(height=256,
                         width=256),
                A.CoarseDropout(min_holes=int(256*256*0.89),
                                max_holes=int(256*256*0.91),
                                max_width=1,
                                max_height=1,
                                p=1),
                A.VerticalFlip(),
                ToTensorV2()
            ]
        )
    
    def _split_data(self):
        raise NotImplementedError("Dataset._split_data for split to train and valid datasets")

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        img_path, depth_path = (os.path.join(self._data_path_, item) for item in self.anno.row(idx))
        img, depth = Image.open(img_path), Image.open(depth_path)
        raise NotImplementedError("dataset.__getitem__")


if __name__ == "__main__":
    dataset = NYUDataset(stage="train")
    data = dataset[0]

