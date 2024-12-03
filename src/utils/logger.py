import os
import time
import shutil
import typing as t



class Logger:
    def __init__(self, trainer):
        self.rank = trainer.meta["rank"]
        self.world_size = trainer.meta["world_size"]
        self.time = time.time()
        self.lr = trainer.meta["params"]["learning_rate"]
        self.num_epochs = trainer.meta["params"]["max_epoch"]
        self.total_iters = self.num_epochs * (len(trainer.train_loader) + len(trainer.valid_loader))
        self.curr_time = time.time()
        curr_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime())
        message =  f"Начало обучения. Текущее время: {curr_time}. "
        message += f"Количество эпох: {self.num_epochs}. Стартовая скорость обучения: {self.lr}"

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
                    stage_dict = type(target_list)()

    def finish(self):
        self.progress = 100.0

    def _print_message(self, message: str, last_iter: bool):
        terminal_width = shutil.get_terminal_size().columns
        if len(message) > terminal_width:
            message = message[:terminal_width]
        print(" " * terminal_width, end="\r")
        print(message, end = "" if last_iter else "\r")

    def _get_losses(self, losses_dict, last_iter=False):
        loss_message = str()
        if len(losses_dict) == 0:
            loss_message = "None"
        else:
            for loss_name, loss in losses_dict.items():
                if last_iter:
                    l = sum(loss) / len(loss) / self.world_size
                    losses_dict[loss_name] = [losses_dict[loss_name][-1]]
                else:
                    l = loss[-1] / self.world_size
                loss_message += f"{loss_name}: {l:.4} "
        return loss_message[:-1]
    
    def step(self, data: t.Dict[str, t.Dict[str, t.List[float]]], stage: str = "train"):
        self.iters[stage] += 1
        self.iters["epoch"] += 1
        self.iters["total"] += 1
        self.progress = self.iters["total"] / self.total_iters * 100
        for t_type, target_dict in data.items():
            for t_name, target_float in data[t_type].items():
                self.accum_dicts[stage][t_type][t_name].append(target_float)
        for target_type, target_dict in data.items():
            raise NotImplementedError("need to get message....") # TODO
            self._get_losses(target_dict)
        self._iter_message()

    def get_message(self, _dict, last_iter=False):
        message = str()
        if len(_dict) == 0:
            message = "None"
        else:
            for name, values in _dict.items():
                if last_iter:
                    l = sum(values) / len(values)
                    _dict[name] = [_dict[name][-1]]
                else:
                    l = values[-1]
                message += f"{name}: {l:.4}"
        return message

    def graph_write(self, name, value, info, last_iter=False):
        raise NotImplementedError("Logger.graph_write")

    def rank_0(self, call):
        if self.rank == 0:
            return call
        else:
            return None

