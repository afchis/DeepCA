

def get_model(params):
    if params["model"]["name"].startswith("resnet"):
        from .model_heads import DeepCAResNet
        model = DeepCAResNet(params)
    else:
        raise NotImplementedError(f"model: {params['model']['name']} is not implemented")
    return model


