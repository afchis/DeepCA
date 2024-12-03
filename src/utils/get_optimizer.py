import torch.optim as optim


def get_optimizer(model, params):
    optim_list = ["SGD", "Adam"]
    assert params["solver"] in optim_list, ("wrong optimizer name in config. Please choice", optim_list)
    if params["solver"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])
    elif params["solver"] == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=params["learning_rate"],
                               betas=(params["beta1"], params["beta2"]),
                               eps=params["eps"],
                               weight_decay=params["weight_decay"])
    else:
        raise NotImplementedError(f"optimizer: {params['solver']} is not implemented")
    return optimizer


