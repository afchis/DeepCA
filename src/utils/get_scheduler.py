import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, params):
    sched_list = ["StepLR", "MultistepLR"]
    assert params["policy"] in sched_list, ("wrong scheduler name in config. Please choice", sched_list)
    if params["policy"] == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])
    elif params["policy"] == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=params["milestones"], gamma=params["gamma"])
    else:
        raise NotImplementedError(f"scheduler: {params['policy']} is not implemented")
    return scheduler


