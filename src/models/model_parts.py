import torch # TEMP
import torch.nn as nn

from .model_blocks import BasicBlock, BottleneckBlock, ConvBlock, DeConvBlock


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels, block, layers, admm=False, rho=1, iters=20):
        super().__init__()
        self.block = block
        self.num_layers = layers
        self.ch_nums = [64, 128, 256, 512]
        self.admm = admm
        self.rho = rho
        self.iters = 20
        self.in_ch = 64
        kernel_size = 3 if len(layers) < 3 else 7
        _layers_ = list()
        _layers_.append(nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if len(layers) > 2 else None
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
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0 and self.pool:
                x = self.pool(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, channels=[512, 256, 128, 64], admm=False, rho=1, iters=20):
        super().__init__()
        blocks = list()
        channels.append(channels[-1]//2)
        ch = channels[0]
        blocks.append(nn.Sequential(
            DeConvBlock(ch, ch, admm=admm, rho=rho, iters=iters),
            ConvBlock(ch, ch, kernel_size=3, stride=1,
                      admm=admm, rho=rho, iters=iters)
        ))
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i+1]
            blocks.append(nn.Sequential(
                DeConvBlock(in_ch, out_ch, admm=admm, rho=rho, iters=iters),
                ConvBlock(out_ch, out_ch, kernel_size=3, stride=1,
                          admm=admm, rho=rho, iters=iters)
            ))
        if len(channels) < 4:
            blocks = blocks[:-1]
            channels = channels[:-1]
        blocks.append(nn.Conv2d(channels[-1], 1, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


def resnet18(in_channels=4, admm=False, rho=1, iters=20, num_layers=None):
    layers = [2, 2, 2, 2]
    if num_layers: layers = layers[:num_layers]
    return ResNetBackbone(in_channels, block=BasicBlock, layers=layers,
                          admm=admm, rho=rho, iters=iters)


def resnet50(in_channels=4, admm=False, rho=1, iters=20, num_layers=None):
    layers = [3, 4, 6, 4]
    if num_layers: layers = layers[:num_layers]
    return ResNetBackbone(in_channels, block=BottleneckBlock, layers=layers,
                          admm=admm, rho=rho, iters=iters)


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


