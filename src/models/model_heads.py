import torch # TEMP
import torch.nn as nn

from .model_blocks import ConvBlock
from .resnet import resnet18, resnet50


class DeepCAResNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        backbone_mapping = {
            "resnet18": resnet18,
            "resnet50": resnet50
        }
        get_backbone = backbone_mapping[params["model"]["name"]]
        in_channels = params["model"]["in_channels"]
        admm = params["model"]["admm"]
        rho = params["model"]["rho"] if admm else None
        iters = params["model"]["iters"] if admm else None
        self.backbone = get_backbone(in_channels=in_channels, admm=admm, rho=rho, iters=iters)
        self.decoder = NotImplemented

    def forward(self, x):
        x = self.backbone(x)
        raise NotImplementedError("class DeepCAResNet.forward()")


class DeepCAUnet(nn.Module):
    def __init__(self, in_channels=4, admm=False, rho=1, iters=20):
        super().__init__()
        self.down1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, admm=admm, rho=rho, iters=iters),
            DCAConvBlock(32, 32, kernel_size=3, stride=2, admm=admm, rho=rho, iters=iters),
        )
        self.down2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=1, admm=admm, rho=rho, iters=iters),
            ConvBlock(64, 64, kernel_size=3, stride=2, admm=admm, rho=rho, iters=iters),
        )
        self.down3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1, admm=admm, rho=rho, iters=iters),
            ConvBlock(128, 128, kernel_size=3, stride=2, admm=admm, rho=rho, iters=iters),
        )
        self.middle = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=1, admm=admm, rho=rho, iters=iters),
            ConvBlock(256, 128, kernel_size=3, stride=1, admm=admm, rho=rho, iters=iters),
        )
        self.up3 = nn.Sequential()
        self.up2 = nn.Sequential()
        self.up1 = nn.Sequential()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        mid = self.middle(x3)
        raise NotImplementedError("class DeepCAUnet.forward()")


if __name__ == "__main__":
    x = torch.rand([1, 4, 256, 256])
    model = DeepCAResNet(in_channels=4)
    out = model(x)


