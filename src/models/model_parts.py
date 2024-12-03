import torch # TEMP
import torch.nn as nn

from .model_blocks import BasicBlock, BottleneckBlock


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels, block, layers, admm=False, rho=1, iters=20):
        super().__init__()
        self.ch_nums = [64, 128, 256, 512]
        self.admm = admm
        self.rho = rho
        self.iters = 20
        self.in_ch = 64
        _layers_ = list()
        _layers_.append(nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ))
        for i, l in enumerate(layers):
            stride = 1 if i == 0 else 2
            _layers_.append(self._make_layers(block, self.ch_nums[i], l, stride))
        self.layers = nn.Sequential(*_layers_)

    def _make_layers(self, block, out_ch, num_blocks, stride=1):
        if stride != 1 or self.in_ch != out_ch * block.expantion:
            downsample = nn.Sequential(nn.Conv2d(in_channels=self.in_ch,
                                                 out_channels=out_ch*block.expantion,
                                                 kernel_size=1,
                                                 stride=stride,
                                                 bias=False),
                                       nn.BatchNorm2d(out_ch*block.expantion),)
        else:
            downsample = None
        layers = list()
        layers.append(block(self.in_ch, out_ch, stride, downsample, self.admm, self.rho, self.iters))
        self.in_ch = out_ch * block.expantion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def resnet18(in_channels=4, admm=False, rho=1, iters=20):
    return ResNetBackbone(in_channels, block=BasicBlock, layers=[2, 2, 2, 2], admm=admm, rho=rho, iters=iters)


def resnet50(in_channels=4, admm=False, rho=1, iters=20):
    return ResNetBackbone(in_channels, block=BottleneckBlock, layers=[3, 4, 6, 4], admm=admm, rho=rho, iters=iters)


if __name__ == "__main__":
    x = torch.rand([1, 4, 256, 256])

    model = resnet18(in_channels=4, admm=False)
    out = model(x)
    print("resnet18.admm -> False:", out.shape)

    model = resnet18(in_channels=4, admm=True, rho=1, iters=20)
    out = model(x)
    print("resnet18.admm -> True:", out.shape)

    model = resnet50(in_channels=4, admm=False)
    out = model(x)
    print("resnet50.admm -> False:", out.shape)

    model = resnet50(in_channels=4, admm=True, rho=1, iters=20)
    out = model(x)
    print("resnet50.admm -> True:", out.shape)


