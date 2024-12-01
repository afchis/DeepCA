import torch # TEMP
import torch.nn as nn


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels=4, block, layers):
        super().__init__()
        self.in_ch = 64
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layers(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layers(block, 512, layers[3], stride=2)

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
        layers.append(block(self.in_ch, out_ch, stride, downsample))
        self.in_ch = out_ch * block.expantion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


if __name__ == "__main__":
    x = torch.rand([1, 4, 256, 256])
    resnet = ResNetBackbone(in_channels=4, block=BottleneckBlock, layers=[3, 4, 6, 3])
    out = resnet(x)


