
# code partially modified from https://github.com/ultralytics/ultralytics


from typing import Any
import torch
from torch import nn
import torchvision


class Conv(nn.Module):
    # basic Conv block used in YOLOv8
    def __init__(self, c_in, c_out, k=3, s=2, p=1, ):
        super(Conv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)
    
    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return super().__call__(*args, **kwds)


class Bottleneck(nn.Module):
    # Bottleneck block used in YOLOv8
    def __init__(self, c, shortcut=True,):
        super(Bottleneck, self).__init__()
        self.shortcut = shortcut

        self.net = nn.Sequential(
            Conv(c, c, 3, 1, 1),
            Conv(c, c, 3, 1, 1),
        )

    def forward(self, x) -> torch.Tensor:
        if self.shortcut:
            return x + self.net(x)
        else:
            return self.net(x)


class C2f(nn.Module):
    # 2D convolutional feature extractor
    def __init__(self, c_in, c_out, n=1, shortcut=True):
        super(C2f, self).__init__()
        self.c_hid = c_out // 2

        self.conv1 = Conv(c_in, c_out, 1, 1, 0)
        self.conv2 = Conv((2+n)*self.c_hid, c_out, 1, 1, 0)
        self.ms = nn.ModuleList(
            [Bottleneck(self.c_hid, shortcut) for _ in range(n)]
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
        self.conv1 = Conv(c_in, c_hid, 1, 1, 0)
        self.conv2 = Conv(c_hid * 4, c_out, 1, 1, 0)
        self.m = nn.MaxPool2d(k, 1, k//2)

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y = self.conv2(torch.cat((x, y1, y2, y3), dim=1))
        return y


class Backbone(nn.Module):
    # YOLO-like backbone
    def __init__(self, depth, width, ratio):
        super(Backbone, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio

        self.pre = nn.Sequential(
            # 3*640*640 -> w*320*320
            Conv(3, width, 3, 2, 1),
            # w*320*320 -> 2w*160*160
            Conv(width, width*2, 3, 2, 1),
            # 2w*160*160 -> 2w*80*80
            C2f(width*2, width*2, n=depth),
            # 2w*80*80 -> 4w*40*40
            Conv(width*2, width*4, 3, 2, 1),
        )
        # 4w*40*40 -> 4w*40*40
        self.c4 = C2f(width*4, width*4, n=depth*2)
        # 4w*40*40 -> 8w*20*20
        self.c5 = Conv(width*4, width*8, 3, 2, 1)
        # 8w*20*20 -> 8w*20*20
        self.c6 = C2f(width*8, width*8, n=depth*2)
        # 8w*20*20 -> 8wr*20*20
        self.c7 = Conv(width*8, width*8*ratio, 3, 2, 1)
        # 8wr*20*20 -> 8wr*20*20
        self.c8 = C2f(width*8*ratio, width*8*ratio, n=depth)
        # 8wr*20*20 -> 8wr*20*20
        self.c9 = SPPF(width*8*ratio, width*8*ratio)


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
    # YOLO-like feature pyramid network
    def __init__(self, depth, width, ratio):
        super(FPN, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio

        # from batch*channels*20*20 to batch*channels*40*40
        self.up1 = nn.Upsample(scale_factor=2)
        self.c12 = C2f(width*8*(1+ratio), width*8, shortcut=False)
        self.up2 = nn.Upsample(scale_factor=2)
        self.c15 = C2f(width*12, width*4, shortcut=False)

        self.c16 = Conv(width*4, width*4, 3, 2, 1)
        self.c18 = C2f(width*12, width*8, n=depth, shortcut=False)

        self.c19 = Conv(width*8, width*8, 3, 2, 1)
        self.c21 = C2f(width*8*(1+ratio), width*8*ratio, n=depth, shortcut=False)

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


class ConvFusion(nn.Module):
    # Convolutional fusion
    def __init__(self, c_in, c_out, k=3, s=2, p=1):
        super(ConvFusion, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )

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

        # self.conv4 = Conv(width*4*2, width*4, 3, 1, 1)
        self.conv4 = ConvFusion(width*4*2, width*4, 3, 1, 1)
        # self.conv6 = Conv(width*8*2, width*8, 3, 1, 1)
        self.conv6 = ConvFusion(width*8*2, width*8, 3, 1, 1)
        # self.sppf9 = Conv(width*8*ratio*2, width*8*ratio, 3, 1, 1)
        self.sppf9 = ConvFusion(width*8*ratio*2, width*8*ratio, 3, 1, 1)

    def forward(self, c4_1, c6_1, sppf_1, c4_2, c6_2, sppf_2) -> torch.Tensor:
        # c4 = torch.cat([c4_1, c4_2], dim=1)
        c4 = self.conv4(c4_1, c4_2)
        # c6 = torch.cat([c6_1, c6_2], dim=1)
        c6 = self.conv6(c6_1, c6_2)
        # sppf = torch.cat([sppf_1, sppf_2], dim=1)
        sppf = self.sppf9(sppf_1, sppf_2)

        return c4, c6, sppf


class Detect(nn.Module):
    # FCOS-like detector
    def __init__(self, c, num_class):
        super(Detect, self).__init__()
        self.c = c # channels
        self.num_class = num_class


        self.branch1 = nn.Sequential(
            Conv(c, c, 3, 1, 1),
            Conv(c, c, 3, 1, 1),
        )
        self.classifier = nn.Conv2d(c, num_class, 1, 1, 0)

        self.centerness = nn.Conv2d(c, 1, 1, 1, 0)

        self.regressor = nn.Sequential(
            Conv(c, c, 3, 1, 1),
            Conv(c, c, 3, 1, 1),
            nn.Conv2d(c, 4, 1, 1, 0),
        )

    def forward(self, x) -> 'tuple[torch.Tensor]':
        b1 = self.branch1(x)
        cls = self.classifier(b1)
        ctr = self.centerness(b1)
        bbox = self.regressor(x)

        return cls, ctr, bbox
    

class Head(nn.Module):
    # YOLO-like head
    def __init__(self, depth, width, ratio, num_class, reg_max):
        super(Head, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio
        self.num_class = num_class
        self.reg_max = reg_max

        self.fpn = FPN(depth, width, ratio)
        self.detector1 = Detect(width*4, reg_max, num_class)
        self.detector2 = Detect(width*8, reg_max, num_class)
        self.detector3 = Detect(width*8*ratio, reg_max, num_class)


    def forward(self, c4, c6, sppf) -> 'tuple[torch.Tensor]':
        c15, c18, c21 = self.fpn(c4, c6, sppf)
        cls1, bbox1 = self.detector1(c15)
        cls2, bbox2 = self.detector2(c18)
        cls3, bbox3 = self.detector3(c21)
        return cls1, bbox1, cls2, bbox2, cls3, bbox3


class YNet(nn.Module):
    @staticmethod
    def mini_ynet():
        # base sizes
        # equivalent to 1/3, 1/4, 2
        depth, width, ratio = 1, 16, 2
        # depth = [1, 2, 2, 1, 1, 1, 1, 1]
        # width = [3, 16, 32, 64, 128, 192]
        # ratio = [1, ]
        return YNet(depth, width, ratio)

    def __init__(self, depth, width, ratio, num_class=20, reg_max=20):
        super(YNet, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio
        self.num_class = num_class
        self.reg_max = reg_max

        self.backbone1 = Backbone(depth, width, ratio)
        self.backbone2 = self.backbone1
        self.fusor = FusionPlant(depth, width, ratio)
        self.head = Head(depth, width, ratio, num_class, reg_max)

    def forward(self, x1, x2) -> 'tuple[torch.Tensor]':
        c4_1, c6_1, sppf_1 = self.backbone1(x1)
        c4_2, c6_2, sppf_2 = self.backbone2(x2)
        c4, c6, sppf = self.fusor(c4_1, c6_1, sppf_1, c4_2, c6_2, sppf_2)
        cls1, bbox1, cls2, bbox2, cls3, bbox3 = self.head(c4, c6, sppf)
        return cls1, bbox1, cls2, bbox2, cls3, bbox3
    

class TinyYNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.extractor1 = nn.Sequential(
            # 3*640*640 -> 16*320*320
            Conv(3, 16, 3, 2, 1),
            # 16*320*320 -> 32*160*160
            Conv(16, 32, 3, 2, 1),
            # 32*160*160 -> 64*80*80
            Conv(32, 64, 3, 2, 1),
        )
        self.extractor2 = nn.Sequential(
            Conv(3, 16, 3, 2, 1),
            Conv(16, 32, 3, 2, 1),
            Conv(32, 64, 3, 2, 1),
        )

        self.fusor = ConvFusion(64*2, 64, 3, 2, 1)

        self.detector = Detect(64, 20)


    def forward(self, x1, x2) -> 'tuple[torch.Tensor]':
        f1 = self.extractor1(x1)
        f2 = self.extractor2(x2)

        f = self.fusor(f1, f2)
        cls, str, box = self.detector(f)

        return cls, str, box
    
    def predict(self, x1, x2) -> 'tuple[torch.Tensor]':
        assert NotImplementedError
        cls, ctr, box = self.forward(x1, x2)
        torchvision.ops.nms(box, ctr, 0.5)
        return cls, box


class YOLOv8(nn.Module):
    def __init__(self, depth, width, ratio, num_class=20, reg_max=20):
        super().__init__()
        self.backbone = Backbone(depth, width, ratio)
        self.head = Head(depth, width, ratio, num_class, reg_max)
        self.detector1 = Detect(width*4, 8, reg_max, num_class)
        self.detector2 = Detect(width*8, 16,  reg_max, num_class)
        self.detector3 = Detect(width*8*ratio, 32, reg_max, num_class)

    def forward(self, x1) -> 'tuple[torch.Tensor]':
        c4, c6, sppf = self.backbone(x1)
        cls1, bbox1, cls2, bbox2, cls3, bbox3 = self.head(c4, c6, sppf)
        return cls1, bbox1, cls2, bbox2, cls3, bbox3



class DetectLoss():
    def __init__(self) -> None:
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.focal = torchvision.ops.sigmoid_focal_loss
    
    def encode(self, target: torch.Tensor, c: int, img_size: 'tuple[int]'):
        """
        Encode target to cls, ctr, bbox for loss calculation
        target: x0, y0, x1, y1, cls_idx
        c: channels (num_classes)
        s: stride
        """
        batch = target.shape[0]
        w, h = img_size
        cls = torch.zeros((batch, c, w, h))
        ctr = torch.zeros((batch, 1, w, h))
        bbox = torch.zeros((batch, 4, w, h))

        for i in range(batch):
            x0, y0, x1, y1, cls_idx = target[i]
            cls_idx = int(cls_idx)
            cls[i, cls_idx, x0:x1, y0:y1] = 1
            for x in range(x0, x1):
                for y in range(y0, y1):
                    # bbox
                    l, t, r, b = (x-x0) / w, (y-y0) / h, (x1-x) / w, (y1-y) / h
                    bbox[i, :, x, y] = torch.tensor([l, t, r, b])
                    # ctr
                    ctr_val = (min(l, r) * min(t, b)) / (max(l, r) * max(t, b))
                    ctr[i, 0, x, y] = torch.sqrt(ctr_val)

        return cls, ctr, bbox

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, ) -> torch.Tensor:
        # pred: (cls, ctr, bbox)
        # target: (batch, (x0, y0, x1, y1, cls))
        # cls: (batch, num_class, h, w)
        # ctr: (batch, 1, h, w)
        # bbox: (batch, 4, h, w)

        batch = pred[0].shape[0]
        channels = pred[0].shape[1]
        image_size = pred[0].shape[2:]
        cls_gt, ctr_gt, bbox_gt = self.encode(target, channels, image_size)

        cls_loss = self.focal(pred[0], cls_gt).mean()
        ctr_loss = self.bce(pred[1], ctr_gt)
        # bbox_loss where cls_gt > 0
        bbox_loss = 0
        for i in range(batch):
            if cls_gt[i].sum() > 0:
                bbox_loss += self.mse(pred[2][i], bbox_gt[i])
        loss = cls_loss + ctr_loss + bbox_loss
        return loss



def test():
    net = TinyYNet()
    x1 = torch.randn(32, 3, 640, 640)
    x2 = torch.randn(32, 3, 640, 640)
    
    cls, ctr, bbox = net(x1, x2)
    loss_fn = DetectLoss()
    loss = loss_fn((cls, ctr, bbox), torch.randint(-1, 1, (32, 5)))
    loss.backward()
    print(loss.item())

def test_yolo():
    net = YOLOv8(1, 16, 1)
    x1 = torch.randn(32, 3, 640, 640)
    cls1, bbox1, cls2, bbox2, cls3, bbox3 = net(x1)
    print(cls1.shape, bbox1.shape, cls2.shape, bbox2.shape, cls3.shape, bbox3.shape)

    param_num = 0
    for param in net.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print("Total number of parameters: {} M".format(param_num / 1e6))


if __name__ == '__main__':
    test()
    # test_yolo()