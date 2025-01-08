import os
import time
import shutil
import typing as t

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, trainer):
        self.rank = trainer.meta["rank"]
        self.world_size = trainer.meta["world_size"]
        self.time = time.time()
        self.lr = trainer.meta["params"]["learning_rate"]
        self.num_epochs = trainer.meta["params"]["max_epoch"]
        self.total_iters_in_epoch = len(trainer.train_loader) + len(trainer.valid_loader)
        self.total_iters = self.num_epochs * self.total_iters_in_epoch
        self.curr_time = time.time()
        curr_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime())
        message =  f"Learning start. Currnet time: {curr_time}. "
        message += f"Num epoch: {self.num_epochs}. Start lr: {self.lr}"
        self._print_message(message)
        self.tensorboard_writter = SummaryWriter(f"_logs_/{trainer.meta['params']['experiment_name']}/logs")

    def init(self, loss_metrics: t.Dict[str, t.Dict[str, t.List[str]]]):
        self.progress = 0.0
        self.epoch = 0
        self.iters = {
            "total": 0,
            "epoch": 0,
            "train": 0,
            "valid": 0,
        }
        self.accum_dicts = dict()
        for stage in loss_metrics.keys():
            self.accum_dicts[stage] = dict()
            for target_type in loss_metrics[stage].keys():
                self.accum_dicts[stage][target_type] = dict()
                for target_name in loss_metrics[stage][target_type]:
                    self.accum_dicts[stage][target_type][target_name] = list()

    def new_epoch(self):
        self.epoch += 1
        self.iters["epoch"] = 0
        self.iters["train"] = 0
        self.iters["valid"] = 0
        for stage, stage_dict in self.accum_dicts.items():
            for t_type, target_dict in stage_dict.items():
                for t_name, target_list in target_dict.items():
                    self.accum_dicts[stage][t_type][t_name] = []

    def finish(self):
        self.progress = 100.0

    def _print_message(self, message: str, last_iter: bool = True):
        terminal_width = shutil.get_terminal_size().columns
        if len(message) > terminal_width:
            message = message[:terminal_width]
        print(" " * terminal_width, end="\r")
        print(message, end = "" if last_iter else "\r")

    def step(self, data: t.Dict[str, t.Dict[str, t.List[float]]], stage: str = "train"):
        self.iters[stage] += 1
        self.iters["epoch"] += 1
        self.iters["total"] += 1
        self.progress = self.iters["total"] / self.total_iters * 100
        for t_type, target_dict in self.accum_dicts[stage].items():
            for t_name, target_float in data[t_type].items():
                self.accum_dicts[stage][t_type][t_name].append(target_float)
        self._iter_message()

    def _iter_message(self):
        stage_messages = list()
        for stage, stage_dict in self.accum_dicts.items():
            type_message = list()
            for t_type, target_dict in stage_dict.items():
                target_message = str()
                for t_name, target_list in target_dict.items():
                    if not target_list:
                        last_iter = False
                        value_str = "None"
                    elif self.iters["epoch"] == self.total_iters_in_epoch:
                        last_iter = True
                        value = sum(target_list) / (len(target_list) * self.world_size)
                        value_str = f"{value:.4f}"
                    else:
                        last_iter = False
                        value = target_list[-1] / self.world_size
                        self.graph_write(stage, t_type, t_name, value)
                        value_str = f"{value:.4f}"
                    target_message += f"{t_name}: {value_str} "
                type_message.append(f"{t_type} -> {target_message}")
            type_message = "| ".join(type_message)
            stage_messages.append(f"{stage}: {type_message}")
        if not last_iter:
            message = f"Train: [{self.progress:.2f}{chr(37)}] Iter: [{self.iters['total']}] || {'|| '.join(stage_messages)}||"
        else:
            message = f"Epoch: {self.epoch} || {'|| '.join(stage_messages)} ||"
        self._print_message(message=message, last_iter=last_iter)

    def graph_write(self, stage: str, target_type: str, target_name: str, value: float):
        name = "_".join((stage, target_name))
        self.tensorboard_writter.add_scalars(target_type, {name: value}, self.iters["train"])

    def rank_0(self, call):
        if self.rank == 0:
            return call
        else:
            return None

