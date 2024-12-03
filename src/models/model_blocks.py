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
    def __init__(self, in_ch, out_ch, kernel_size=3,
                 stride=1, admm=False, rho=1, iters=20):
        super().__init__()
        self.admm = admm
        self.rho = rho
        self.iters = iters
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=kernel_size//2)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def _step_forward(self, x):
        return self.norm(self.conv(x))

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
    def __init__(self, in_ch, out_ch, admm=False, rho=1, iters=20):
        super().__init__()
        self.admm = admm
        self.rho = rho
        self.iters = iters
        self.deconv = nn.ConvTranspose2d(in_channels=in_ch,
                                         out_channels=out_ch,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def _step_forward(self, x):
        return self.norm(self.deconv(x))

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


if __name__ == "__main__":
    x = torch.rand([1, 3, 32, 32])
    print("x.shape -> ", x.shape)

    block = ConvBlock(3, 16, 3, stride=2, admm=False)
    out = block(x)
    print("ConvBlock.admm -> False:", out.shape)

    block = ConvBlock(3, 16, 3, stride=2, admm=True, rho=1, iters=20)
    out = block(x)
    print("ConvBlock.admm -> False:", out.shape)

    x = torch.rand([1, 128, 4, 4])
    print("x.shape -> ", x.shape)

    block = DeConvBlock(128, 64, admm=False)
    out = block(x)
    print("DeConvBlock.admm -> False:", out.shape)

    block = DeConvBlock(128, 64, admm=True, rho=1, iters=20)
    out = block(x)
    print("DeConvBlock.admm -> False:", out.shape)

