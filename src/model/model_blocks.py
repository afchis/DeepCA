import torch
import torch.nn as nn



class BottleneckBlock(nn.Module):
    expantion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, rho=1, iters=20):
        super().__init__()
        self.rho = rho
        self.iters = iters
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expantion, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_ch * self.expantion)
        self.act = nn.ReLU()

    def _step_forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return out
    
    def forward(self, x):
        w = self._step_forward(x)
        for i in range(self.iters):
            w_prev = w.clone()
            w = torch.clamp(w_prev + self.rho * (self.act(w_prev) - w_prev), min=0.)
        return w


class ADNNTDeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_iters: int = 10
    ):
        super().__init__()
        self.num_iters = num_iters
        self.conv = nn.Conv2dTranspose(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       padding=kernel_size//2,
                                       stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        w = x
        for _ in range(self.num_iters):
            w = torch.clamp(self.norm(self.conv(w)), min=0)
        return x


class ADNNConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_iters: int = 10
    ):
        super().__init__()
        self.num_iters = num_iters
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size//2,
                              stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        w = x
        for _ in range(self.num_iters):
            w = torch.clamp(self.norm(self.conv(w)), min=0)
        return x


class DefaultConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size//2,
                              stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


if __name__ == "__main__":
    block = ADNNConvBlock(3, 16, 3)
    x = torch.rand([1, 3, 32, 32])
    out = block(x)

