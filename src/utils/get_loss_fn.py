import torch.nn as nn


def get_loss_fn(params):
    crit_list = ["l1_loss", "mse_loss"]
    assert params["criterion"] in crit_list, ("wrong ctiretion name in config. Please choice", optim_list)
    if params["criterion"] == "l1_loss":
        criterion = nn.L1Loss()
    elif params["criterion"] == "mse_loss":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f"criterion: {params['criterion']} is not implemented")
    return criterion

