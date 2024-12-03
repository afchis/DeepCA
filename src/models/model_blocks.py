import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expantion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, admm=False, rho=1, iters=20):
        super().__init__()
        self.admm = admm
        self.rho = rho
        self.iters = iters
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def _step_forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        w = self._step_forward(x)
        if self.admm:
            for _ in range(self.iters):
                w_prev = w.clone()
                w = w_prev + (self.rho / (self.rho + 1)) * (self.act(w_prev) - w_prev)
                w = torch.clamp(w, min=0.)
        else:
            w = self.act(w)
        return w


class BottleneckBlock(nn.Module):
    expantion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, admm=False, rho=1, iters=20):
        super().__init__()
        self.admm = admm
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
        if self.admm:
            for _ in range(self.iters):
                w_prev = w.clone()
                w = w_prev + (self.rho / (self.rho + 1)) * (self.act(w_prev) - w_prev)
                w = torch.clamp(w, min=0.)
        else: 
            w = self.act(w)
        return w


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        admm: bool = False,
        rho: int = 1,
        iters: int = 20
    ):
        super().__init__()
        self.admm = admm
        self.rho = rho
        self.iters = iters
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size//2,
                              stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size//2)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def _step_forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x

    def forward(self, x):
        w = self._step_forward(x)
        if self.admm:
            for _ in range(self.iters):
                w_prev = w.clone()
                w = w_prev + (self.rho / (self.rho + 1)) * (self.act(w_prev) - w_prev)
                w = torch.clamp(w, min=0.)
        else:
            w = self.act(w)
        return w


class DeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        admm: bool = True,
        rho: int = 1,
        iters: int = 20
    ):
        super().__init__()
        self.iters = iters
        self.deconv1 = nn.Conv2dTranspose(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          padding=kernel_size//2,
                                          stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.deconv2 = nn.Conv2dTranspose(in_channels=out_channeos,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          padding=kernel_size//2)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def _step_forward(self, x):
        x = self.act(self.norm1(self.deconv1(x)))
        return self.norm2(self.deconv2(x))


    def forward(self, x):
        w = self._step_forward(x)
        for _ in range(self.iters):
            w_prev = w.clone()
            w = w_prew + (self.rho / (self.rho + 1)) * (self.act(w_prev) - w_prev)
            w = torch.clamp(w, min=0.)
        return x


if __name__ == "__main__":
    x = torch.rand([1, 3, 32, 32])

    block = ConvBlock(3, 16, 3, stride=2, admm=False)
    out = block(x)
    print("ConvBlock.admm -> False:", out.shape)

    block = ConvBlock(3, 16, 3, stride=2, admm=True, rho=1, iters=20)
    out = block(x)
    print("ConvBlock.admm -> False:", out.shape)


