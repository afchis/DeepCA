from torch.utils.data import DataLoader

from .dataset import NYUDataset


def get_loader(params, stage):
    dataset = NYUDataset(stage)
    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        shuffle=True if stage == "train" else False
    )
    return dataloader


