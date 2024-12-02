import torch # TEMP
import torch.nn as nn

from .model_blocks import DCAConvBlock
from .resnet import ResNetBackbone, BottleneckBlock


class DeepCAResNet(nn.Module):
    def __init__(self, in_channels=4, block=BottleneckBlock, layers=[3, 4, 6, 3]):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels=4, block=block, layers=layers)
        self.decoder = NotImplemented

    def forward(self, x):
        x = self.backbone(x)
        raise NotImplementedError("class DeepCAResNet.forward()")


class DeepCAUnet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.down1 = nn.Sequential(
            DCAConvBlock(3, 32, kernel_size=3, stride=1, num_iters=10),
            DCAConvBlock(32, 32, kernel_size=3, stride=2, num_iters=10),
        )
        self.down2 = nn.Sequential(
            DCAConvBlock(32, 64, kernel_size=3, stride=1, num_iters=10),
            DCAConvBlock(64, 64, kernel_size=3, stride=2, num_iters=10),
        )
        self.down3 = nn.Sequential(
            DCAConvBlock(64, 128, kernel_size=3, stride=1, num_iters=10),
            DCAConvBlock(128, 128, kernel_size=3, stride=2, num_iters=10),
        )
        self.middle = nn.Sequential(
            DCAConvBlock(128, 256, kernel_size=3, stride=1, num_iters=10),
            DCAConvBlock(256, 128, kernel_size=3, stride=1, num_iters=10),
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


