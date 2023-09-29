
# code partially modified from https://github.com/ultralytics/ultralytics

from typing import List, Any

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


class Backbone(nn.Module):
    # YOLO backbone
    def __init__(self, depth, width, ratio):
        super(Backbone, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio

        self.pre = nn.Sequential(
            # 3*640*640 -> w*320*320
            Conv(3, round(64*width), 3, 2, 1),
            # w*320*320 -> 2w*160*160
            Conv(round(64*width), round(128*width), 3, 2, 1),
            # 2w*160*160 -> 2w*80*80
            C2f(round(128*width), round(128*width), n=round(depth*3)),
            # 2w*80*80 -> 4w*40*40
            Conv(round(128*width), round(256*width), 3, 2, 1),
        )
        # 4w*40*40 -> 4w*40*40
        self.c4 = C2f(round(256*width), round(256*width), n=round(depth*6))
        # 4w*40*40 -> 8w*20*20
        self.c5 = Conv(round(256*width), round(512*width), 3, 2, 1)
        # 8w*20*20 -> 8w*20*20
        self.c6 = C2f(round(512*width), round(512*width), n=round(depth*6))
        # 8w*20*20 -> 8wr*20*20
        self.c7 = Conv(round(512*width), round(512*width*ratio), 3, 2, 1)
        # 8wr*20*20 -> 8wr*20*20
        self.c8 = C2f(round(512*width*ratio), round(512*width*ratio), n=round(depth*3))
        # 8wr*20*20 -> 8wr*20*20
        self.c9 = SPPF(round(512*width*ratio), round(512*width*ratio))

    def forward(self, x) -> 'tuple[torch.Tensor]':
        pre = self.pre(x)
        c4 = self.c4(pre)
        p4 = self.c5(c4)
        c6 = self.c6(p4)
        p5 = self.c7(c6)
        c8 = self.c8(p5)
        sppf = self.c9(c8)

        return c4, c6, sppf


class FPN(nn.Module):
    # YOLO feature pyramid network
    def __init__(self, depth, width, ratio):
        super(FPN, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio

        self.up1 = nn.Upsample(scale_factor=2)
        self.c12 = C2f(round(512*width*(1+ratio)), round(512*width), n=round(depth*3), shortcut=False)
        self.up2 = nn.Upsample(scale_factor=2)
        self.c15 = C2f(round(768*width), round(256*width), n=round(depth*3), shortcut=False)

        self.c16 = Conv(round(256*width), round(256*width), 3, 2, 1)
        self.c18 = C2f(round(768*width), round(512*width), n=round(depth*3), shortcut=False)

        self.c19 = Conv(round(512*width), round(512*width), 3, 2, 1)
        self.c21 = C2f(round(512*width*(1+ratio)), round(512*width*ratio), n=round(depth*3), shortcut=False)

    def forward(self, c4, c6, sppf) -> 'tuple[torch.Tensor]':
        up1 = self.up1(sppf)
        cat11 = torch.cat((up1, c6), dim=1)
        c12 = self.c12(cat11)
        up2 = self.up2(c12)
        cat14 = torch.cat((up2, c4), dim=1)
        c15 = self.c15(cat14)

        c16 = self.c16(c15)
        cat17 = torch.cat((c16, c12), dim=1)
        c18 = self.c18(cat17)
        c19 = self.c19(c18)
        cat20 = torch.cat((c19, sppf), dim=1)
        c21 = self.c21(cat20)

        return c15, c18, c21


class Detect(nn.Module):
    # YOLOv8 detectors
    def __init__(self, channels, num_classes, reg_max=8):
        super(Detect, self).__init__()
        self.channels = channels
        self.reg_max = reg_max
        self.num_class = num_classes

        self.num_layers = len(channels)
        self.num_outputs = num_classes + reg_max*4

        self.strides = [8, 16, 32]

        reg_hid_ch = max(channels[0]//4, 16, reg_max*4)
        cls_hid_ch = max(channels[0], min(num_classes, 100))
        self.regressors = nn.ModuleList(
            nn.Sequential(
                Conv(channel, reg_hid_ch, 3,),
                Conv(reg_hid_ch, reg_hid_ch, 3),
                nn.Conv2d(reg_hid_ch, reg_max*4, 1)
                ) for channel in channels
            )
        self.classifiers = nn.ModuleList(
            nn.Sequential(
                Conv(channel, cls_hid_ch, 3),
                Conv(cls_hid_ch, cls_hid_ch, 3),
                nn.Conv2d(cls_hid_ch, num_classes, 1)
                ) for channel in channels
            )
        
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    def forward(self, xs: List[torch.Tensor]) -> 'tuple[torch.Tensor]':
        """
        process the output of FPN
        xs: list[torch.Tensor], output of FPN
        return: list[torch.Tensor], each element is (batch, num_outputs, h, w)
                from each detector
        """
        for i, x in enumerate(xs):
            xs[i] = torch.cat((self.regressors[i](x), self.classifiers[i](x)), dim=1)

        return xs
    
    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return self.forward(*args, **kwds)

      
class Head(nn.Module):
    # YOLO head
    def __init__(self, depth, width, ratio, num_classes, reg_max):
        super(Head, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.fpn = FPN(depth, width, ratio)
        self.detect = Detect(
            [round(256*width), round(512*width), round(512*width*ratio)],
            num_classes,
            reg_max
            )

    def forward(self, c4, c6, sppf) -> 'tuple[torch.Tensor]':
        c15, c18, c21 = self.fpn(c4, c6, sppf)
        detects = self.detect([c15, c18, c21])
        return detects

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        """
        for type hint
        """
        return self.forward(*args, **kwds)

