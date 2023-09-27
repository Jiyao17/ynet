
from typing import Any, Tuple, List

import torch
from torch import nn

from basic import Conv, C2f, SPPF, DFL
from src.utils.tal import make_anchors, dist2bbox


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
        xs: list[torch.Tensor], output of FPN
        """
        shape = xs[0].shape
        for i, x in enumerate(xs):
            xs[i] = torch.cat((self.regressors[i](x), self.classifiers[i](x)), dim=1)

        if self.training:
            return xs
        
        self.anchors_points, self.anchor_strides = \
            (x.transpose(0, 1) for x in make_anchors(xs, self.strides, 0.5))

        x_cat = torch.cat([x.view(shape[0], self.num_outputs, -1) for x in xs], dim=2)
        box, cls = x_cat.split([self.reg_max*4, self.num_class], dim=1)
        dbox = dist2bbox(self.dfl(box), self.anchors_points.unsqueeze(0), True, 1) * self.anchor_strides

        y = torch.cat((dbox, cls.sigmoid()), dim=1)
        
        # shape = (batch, num_outputs, num_anchors)
        # also    (batch, num_classes+4, sum(fmap_shapes[:,0]*fmap_shape[:,1])
        # e.g., fmaps: [80*80, 40*40, 20*20], num_anchors=80*80+40*40+20*20=8400
        # for v8n, shape = (batch, num_classes+4, 8400)
        return y
        

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


class YOLOv8(nn.Module):
    def __init__(self, size: Tuple[float], num_class :int=20, reg_max: int=4):
        super().__init__()
        self.size = size
        self.num_class = num_class
        self.reg_max = reg_max

        depth, width, ratio = size
        self.backbone = Backbone(depth, width, ratio)
        self.head = Head(depth, width, ratio, num_class, reg_max)

    def forward(self, x1) -> Tuple[torch.Tensor]:
        c4, c6, sppf = self.backbone(x1)
        detects = self.head(c4, c6, sppf)
        return detects

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return self.forward(*args, **kwds)


SIZES = {
    'n': (0.33, 0.25, 2.0),
    's': (0.33, 0.50, 2.0),
    'm': (0.50, 0.67, 1.5),
    'l': (1.00, 1.00, 1.0),
    'x': (1.00, 1.25, 1.0),
}



if __name__ == "__main__":
    img = torch.randn(8, 3, 640, 640)
    size = SIZES['n']
    model = YOLOv8(size, 20, 4)
    param_num = sum(p.numel() for p in model.parameters())
    print("param_num:", param_num / 1e6, "M")

    model.train()
    preds = model(img)
    print(preds[0].shape, preds[1].shape, preds[2].shape)

    # yolo = YOLO('../../assets/yolov8n.pt', 'detect')
    # model = yolo.model.DetectionModel()
    # # param_num = sum(p.numel() for p in model.parameters())
    # # print(param_num)
    # summary = torchsummary.summary(model, (3, 224, 224), 1, device='cpu') 
    # print(summary)
    
    # img = torch.randn(1, 3, 640, 640)
    # cls1, bbox1, cls2, bbox2, cls3, bbox3 = yolo(img)
    # print(cls1.shape, bbox1.shape)