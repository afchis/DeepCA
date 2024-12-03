import os
import json
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataloader.get_loader import get_loader
from .models.get_model import get_model
from .utils.get_optimizer import get_optimizer
from .utils.get_scheduler import get_scheduler
# from utils.save_load_model import save_model, load_model
from utils.logger import Logger


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", type=str, default="default.json")
parser_args = parser.parse_args()


class TrainerMultiGPU:
    def __init__(self, params):
        self.meta = {
            "params": params,
            "world_size": int(os.environ["WORLD_SIZE"]),
            "rank": int(os.environ["RANK"])
        }
        self._model_init()
        self._get_loaders()
        self.loss_fn = get_loss_fn(self.meta["params"])
        self.accuracy_helper = accuracy_helper
        self._get_learning_step()
        self.optimizer = get_optim(self.model.parameters(), self.meta["params"])
        self.scheduler = get_scheduler(self.optimizer, self.meta["params"])

    def _model_init(self):
        if len(self.meta["params"]["target_devices"]) == 0:
            self.device = "cpu"
            print(f"Initialization Trainer: CPU")
        else:
            if self.meta["world_size"] == 1:
                self.device = self.meta["params"]["target_device"][0]
                print(f"Initialization Trainer: GPU -> [ device: cuda:{self.device} ]")
            else:
                self.device = self.meta["rank"]
                print(f"Initialization Trainer: MultiGPU -> world_size: " +
                      f"[ {self.meta['world_size']}, rank: {self.meta['rank']} ]")
        # if self.meta["params"]["checkpoint_name"] != "":
        #     model = load_model(self.meta["params"]).to(self.device)
        # else:
        #     raise NotImplementedError("???")
        model = get_model(self.meta["params"]).to(self.device)
        self.model_name = model.__class__.__name__
        if self.meta["world_size"] > 1:
            self.model = DDP(model, device_ids=[self.device])
        else:
            self.model = model

    def _get_loaders(self):
        self.train_loader = get_loader(self.meta["params"], stage="train")
        self.valid_loader = get_loader(self.meta["params"], stage="valid")
        self.train_len = len(self.train_loader)
        self.valid_len = len(self.valid_loader)
        self.valid_ratio = int(self.train_len / self.valid_len)

    def _get_learning_step(self):
        self.logger = Logger(trainer=self)
        loss_metrics = {
            "train": {
                "losses": ["КроссЭнтропия"],
            },
            "valid": {
                "losses": ["КроссЭнтропия"],
            },
        }
        self.logger.init(loss_metrics)
        self.train_step = self._train_step
        self.valid_step = self._valid_step

    @staticmethod
    def setup_single():
        os.environ["WORLD_SIZE"] = "1"
        print("Setup for train on cpu or single gpu: Done.")

    @staticmethod
    def setup(rank, world_size):
        os.environ["RANK"] = str(rank)
        dist.init_process_group(backend="nccl")
        print(f"Setup for Distributed Data Parallel: Rank {rank} -> Done.")

    def _data_to_device(self, data):
        wav, label = data
        wav = [_wav.to(self.device) for _wav in wav]
        label = label.to(self.device)
        return wav, label

    def train(self):
        patience_epochs = self.meta["params"]["patience_epochs"]
        loss_rule = PatienceRule("Loss", patience_epochs, _max=False)
        accuracy_rule = PatienceRule("Accuracy", patience_epochs, _max=True)
        with EarlyStopper(accuracy_rule, loss_rule) as earlystopper:
            for epoch in range(1, self.meta["params"]["max_epoch"] + 1):
                self.logger.new_epoch()
                self._train_epoch(epoch)
                self.scheduler.step()
        self.logger.finish()

    def _train_epoch(self, epoch):
        self.model.train()
        for iter_, data in enumerate(self.train_loader):
            data = self._train_step(data)
            if self.meta["rank"] == 0: self.logger.step(data, stage="train")
            if (iter_ + 1)  % self.valid_ratio == 0:
                data = self._valid_step(data)
                if self.meta["rank"] == 0: self.logger.step(data, stage="valid")
        save_model(self, epoch)

    def _train_step(self, data):
        wav, label = self._data_to_device(data)
        pred = self.model(torch.stack(wav, dim=0))
        loss = self.loss_fn(pred, label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.meta["world_size"] > 1: dist.all_reduce(loss)
        out = {
            "losses": {
                "КроссЭнтропия": loss.item()
            }
        }
        return out

    @torch.no_grad()
    def _valid_step(self, data):
        self.model.eval()
        data = next(iter(self.valid_loader))
        wav, label = self._data_to_device(data)
        pred = list()
        for _wav in wav:
            pred.append(self._inference(_wav))
        pred = torch.cat(pred, dim=0)
        loss = self.loss_fn(pred, label)
        pos, neg = accuracy_helper(pred, label, self.device)
        accuracy = pos / (pos + neg)
        if self.meta["world_size"] > 1:
            dist.all_reduce(loss)
            for item in accuracy: dist.all_reduce(item)
        self.model.train()
        out = {
            "losses": {
                "КроссЭнтропия": loss.item()
            },
            "metrics": {
                "Точность": accuracy.item()
            }
        }
        return out


def main(rank, world_size, params):
    TrainerMultiGPU.setup(rank, world_size)
    trainer = TrainerMultiGPU(params)
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    with open(os.path.join("params", parser_args.params)) as json_file:
        params = json.load(json_file)
    num_nodes = params["num_nodes"]
    num_gpus = len(params["target_devices"])
    if num_gpus == 0 or num_nodes * num_gpus == 1:
        TrainerMultiGPU.setup_single()
        trainer = TrainerMultiGPU(params)
        trainer.train()
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        world_size = num_nodes * num_gpus
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(main, args=(world_size, params,), nprocs=world_size, join=True)


