
# code partially modified from https://github.com/ultralytics/ultralytics

from typing import List

import torch
from torch import nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # basic Conv block used in YOLOv8
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1, d=1):
        super(Conv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)
    
    def __call__(self, *args, **kwds) -> torch.Tensor:
        return super().__call__(*args, **kwds)


class Bottleneck(nn.Module):
    # Bottleneck block used in YOLOv8
    def __init__(self, c_in, c_out, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(Bottleneck, self).__init__()
        self.add = shortcut and c_in == c_out
        c_hid = int(c_out * e)

        self.net = nn.Sequential(
            Conv(c_in, c_hid, k[0], 1),
            Conv(c_hid, c_out, k[1], 1, g=g),
        )

    def forward(self, x) -> torch.Tensor:
        if self.add:
            return x + self.net(x)
        else:
            return self.net(x)


class C2f(nn.Module):
    def __init__(self, c_in, c_out, n=1, shortcut=True, g=1, e=0.5):
        super(C2f, self).__init__()
        c_hid = int(c_out * e)

        self.conv1 = Conv(c_in, c_out, 1, 1)
        self.conv2 = Conv((2+n)*c_hid, c_out, 1)
        self.ms = nn.ModuleList(
            [Bottleneck(c_hid, c_hid, shortcut, g, k=((3,3), (3,3)), e=1.0) for _ in range(n)]
        )

    def forward(self, x) -> torch.Tensor:
        y = list(self.conv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.ms)
        y = torch.cat(y, dim=1)
        y = self.conv2(y)

        return y


class SPPF(nn.Module):
    def __init__(self, c_in, c_out, k=5) -> None:
        super().__init__()
        
        c_hid = c_in // 2
        self.conv1 = Conv(c_in, c_hid, 1, 1)
        self.conv2 = Conv(c_hid * 4, c_out, 1, 1)
        self.m = nn.MaxPool2d(k, 1, k//2)

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y = self.conv2(torch.cat((x, y1, y2, y3), dim=1))
        return y


class DFL(nn.Module):
    """
    Distribution Focal Loss
    """
    
    def __init__(self, c=16) -> None:
        super().__init__()
        
        self.c = c
        self.conv = nn.Conv2d(c, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c, dtype=torch.float)

        self.conv.weight.data = nn.Parameter(x.view(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, a = x.shape
        x = x.view(b, 4, self.c, a).transpose(2, 1).softmax(1)
        x = self.conv(x).view(b, 4, a)
        return x
