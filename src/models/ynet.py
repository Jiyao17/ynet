
from typing import List, Optional, Tuple, Dict
from collections import OrderedDict

import torch
from torch import nn

from src.models.nn import Conv, Backbone, Head, C2f, DFL
from src.models.yolov8 import YOLOv8


class ConvFusor(nn.Module):
    def __init__(self, fusor, *args):
        super(ConvFusor, self).__init__()
        self.fusor = fusor

        if fusor == 'conv2d':
            self.net = nn.Conv2d(*args)
        elif fusor == 'conv':
            self.net = Conv(*args)
        elif fusor == 'c2f':
            self.net = C2f(*args)
        else:
            raise ValueError('Unknown fusor type: {}'.format(fusor))

    def forward(self, x1, x2) -> torch.Tensor:
        x = torch.cat((x1, x2), dim=1)
        x = self.net(x)
        return x


class FusionPlant(nn.Module):
    # fuse two backbone feature maps
    def __init__(self, depth, width, ratio):
        super(FusionPlant, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio

        self.conv4 = ConvFusor('conv2d', round(256*width*2), round(256*width), 1, 1, 0)
        self.conv6 = ConvFusor('conv2d', round(512*width*2), round(512*width), 1, 1, 0)
        self.sppf9 = ConvFusor('conv2d', round(512*width*ratio*2), round(512*width*ratio), 1, 1, 0)

    def forward(self, c4_1, c6_1, sppf_1, c4_2, c6_2, sppf_2) -> torch.Tensor:
        # c4 = torch.cat([c4_1, c4_2], dim=1)
        c4 = self.conv4(c4_1, c4_2)
        # c6 = torch.cat([c6_1, c6_2], dim=1)
        c6 = self.conv6(c6_1, c6_2)
        # sppf = torch.cat([sppf_1, sppf_2], dim=1)
        sppf = self.sppf9(sppf_1, sppf_2)

        return c4, c6, sppf


class YNet(YOLOv8):

    def __init__(self, size: Tuple[float], num_classes :int=20, reg_max: int=4):
        super().__init__(size, num_classes, reg_max)

        depth, width, ratio = size
        # self.backbone1 = Backbone(depth, width, ratio)
        self.backbone1 = self.backbone
        self.fusor = FusionPlant(depth, width, ratio)

    def forward(self, x) -> 'tuple[torch.Tensor]':
        x1, x2 = x
        c4_1, c6_1, sppf_1 = self.backbone(x1)
        c4_2, c6_2, sppf_2 = self.backbone1(x2)
        c4, c6, sppf = self.fusor(c4_1, c6_1, sppf_1, c4_2, c6_2, sppf_2)
        output = self.head(c4, c6, sppf)

        return output

