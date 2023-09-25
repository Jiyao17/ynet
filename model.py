
# code partially modified from https://github.com/ultralytics/ultralytics




from typing import Dict, List, Optional, Tuple, Any, OrderedDict

import torch
from torch import nn, Tensor
import torchvision
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.fcos import FCOSHead
from torchvision.ops import boxes as box_ops



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
    # YOLO-like feature pyramid network
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
        self.c21 = C2f(round(1024*width), round(512*width*ratio), n=round(depth*3), shortcut=False)

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


class FCOSFPN(nn.Module):
    """
    YOLO-like feature pyramid network
    But all outputs have the same number of channels
    """
    def __init__(self, depth, width, ratio):
        super(FCOSFPN, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio

        self.up1 = nn.Upsample(scale_factor=2)
        self.c12 = C2f(round(512*width*(1+ratio)), round(512*width), n=round(depth*3), shortcut=False)
        self.up2 = nn.Upsample(scale_factor=2)
        self.c15 = C2f(round(768*width), round(512*width), n=round(depth*3), shortcut=False)

        self.pre16 = Conv(round(512*width), round(256*width), 3, 1, 1)
        self.c16 = Conv(round(256*width), round(256*width), 3, 2, 1)
        self.c18 = C2f(round(768*width), round(512*width), n=round(depth*3), shortcut=False)

        self.c19 = Conv(round(512*width), round(512*width), 3, 2, 1)
        self.c21 = C2f(round(512*width*(1+ratio)), round(512*width), n=round(depth*3), shortcut=False)

    def forward(self, c4, c6, sppf) -> 'tuple[torch.Tensor]':
        up1 = self.up1(sppf)
        cat11 = torch.cat((up1, c6), dim=1)
        c12 = self.c12(cat11)
        up2 = self.up2(c12)
        cat14 = torch.cat((up2, c4), dim=1)
        c15 = self.c15(cat14)

        pre16 = self.pre16(c15)
        c16 = self.c16(pre16)
        cat17 = torch.cat((c16, c12), dim=1)
        c18 = self.c18(cat17)
        c19 = self.c19(c18)
        cat20 = torch.cat((c19, sppf), dim=1)
        c21 = self.c21(cat20)

        return c15, c18, c21



class ConvFusor(nn.Module):
    # Convolutional fusion
    def __init__(self, c_in, c_out, n=1):
        super(ConvFusor, self).__init__()
        # the simpler the better!
        # self.net = C2f(c_in, c_out, n, shortcut=False)
        self.net = Conv(c_in, c_out, 3, 1, 1)
        # self.net = nn.Conv2d(c_in, c_out, 3, 1, 1)

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

        self.conv4 = ConvFusor(round(256*width*2), round(256*width))
        self.conv6 = ConvFusor(round(512*width*2), round(512*width))
        self.sppf9 = ConvFusor(round(512*width*ratio*2), round(512*width*ratio))

    def forward(self, c4_1, c6_1, sppf_1, c4_2, c6_2, sppf_2) -> torch.Tensor:
        # c4 = torch.cat([c4_1, c4_2], dim=1)
        c4 = self.conv4(c4_1, c4_2)
        # c6 = torch.cat([c6_1, c6_2], dim=1)
        c6 = self.conv6(c6_1, c6_2)
        # sppf = torch.cat([sppf_1, sppf_2], dim=1)
        sppf = self.sppf9(sppf_1, sppf_2)

        return c4, c6, sppf


class FCOSFusionPlant(nn.Module):
    # fuse two backbone feature maps
    def __init__(self, depth, width, ratio):
        super(FCOSFusionPlant, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio

        # self.conv4 = Conv(width*4*2, width*4, 3, 1, 1)
        self.conv4 = ConvFusor(width*4*2, width*8, 3, 1, 1)
        # self.conv6 = Conv(width*8*2, width*8, 3, 1, 1)
        self.conv6 = ConvFusor(width*8*2, width*8, 3, 1, 1)
        # self.sppf9 = Conv(width*8*ratio*2, width*8*ratio, 3, 1, 1)
        self.sppf9 = ConvFusor(width*8*ratio*2, width*8, 3, 1, 1)

    def forward(self, c4_1, c6_1, sppf_1, c4_2, c6_2, sppf_2) -> torch.Tensor:
        # c4 = torch.cat([c4_1, c4_2], dim=1)
        c4 = self.conv4(c4_1, c4_2)
        # c6 = torch.cat([c6_1, c6_2], dim=1)
        c6 = self.conv6(c6_1, c6_2)
        # sppf = torch.cat([sppf_1, sppf_2], dim=1)
        sppf = self.sppf9(sppf_1, sppf_2)

        return c4, c6, sppf


class Detect(nn.Module):
    # YOLOv8-like detector
    def __init__(self, channels, num_classes, reg_max=8, num_convs=2):
        super(Detect, self).__init__()
        self.channels = channels # channels
        self.reg_max = reg_max
        self.num_class = num_classes

        convs = [ Conv(channels, channels, 3, 1, 1) for _ in range(num_convs) ]
        self.regressor = nn.Sequential(
            *convs,
            nn.Conv2d(channels, reg_max*4, 1, 1, 0),
        )
        convs = [ Conv(channels, channels, 3, 1, 1) for _ in range(num_convs) ]
        self.classifier = nn.Sequential(
            *convs,
            nn.Conv2d(channels, num_classes, 1, 1, 0),
        )

    def forward(self, x) -> 'tuple[torch.Tensor]':
        classes = self.classifier(x)
        boxes = self.regressor(x)

        return classes, boxes


class Head(nn.Module):
    # YOLO-like head
    def __init__(self, depth, width, ratio, num_class):
        super(Head, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio
        self.num_class = num_class

        self.fpn = FPN(depth, width, ratio)
        self.detector1 = Detect(width*4, 8, num_class)
        self.detector2 = Detect(width*8, 16, num_class)
        self.detector3 = Detect(width*8*ratio, 32, num_class)


    def forward(self, c4, c6, sppf) -> 'tuple[torch.Tensor]':
        c15, c18, c21 = self.fpn(c4, c6, sppf)
        cls1, ctr1, bbox1 = self.detector1(c15)
        cls2, ctr2, bbox2 = self.detector2(c18)
        cls3, ctr3, bbox3 = self.detector3(c21)
        return cls1, ctr1, bbox1, cls2, ctr2, bbox2, cls3, ctr3, bbox3


class YNetBackbone(nn.Module):
    def __init__(self, depth=0.67, width=0.75, ratio=1.5):
        super(YNetBackbone, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio
        self.out_channels = round(512 * width)

        self.backbone1 = Backbone(depth, width, ratio)
        # self.backbone2 = Backbone(depth, width, ratio)
        self.backbone2 = self.backbone1

        self.fusor = FusionPlant(depth, width, ratio)

        self.fpn = FCOSFPN(depth, width, ratio)


    def forward(self, x) -> 'tuple[torch.Tensor]':
        c4_1, c6_1, sppf_1 = self.backbone1(x[0])
        c4_2, c6_2, sppf_2 = self.backbone2(x[1])
        c4, c6, sppf = self.fusor(c4_1, c6_1, sppf_1, c4_2, c6_2, sppf_2)
        c15, c18, c21 = self.fpn(c4, c6, sppf)
        features = OrderedDict([
            ("0", c15),
            ("1", c18),
            ("2", c21),
        ])
        # features = OrderedDict([("0", features)])
        return features


class FCOSBackbone(nn.Module):
    # YOLO-like backbone
    def __init__(self, depth, width, ratio):
        super(FCOSBackbone, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio

        self.out_channels = round(512 * width)

        self.backbone = Backbone(depth, width, ratio)
        self.fpn = FCOSFPN(depth, width, ratio)

    def forward(self, x) -> 'tuple[torch.Tensor]':
        c4, c6, sppf = self.backbone(x)
        c15, c18, c21 = self.fpn(c4, c6, sppf)
        features = OrderedDict([
            ("0", c15),
            ("1", c18),
            ("2", c21),
        ])
        return features



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

    def __init__(self, depth, width, ratio, num_class=20):
        super(YNet, self).__init__()
        self.depth = depth
        self.width = width
        self.ratio = ratio
        self.num_class = num_class

        self.backbone1 = Backbone(depth, width, ratio)
        self.backbone2 = self.backbone1
        self.fusor = FusionPlant(depth, width, ratio)
        self.head = Head(depth, width, ratio, num_class)

    def forward(self, x1, x2) -> 'tuple[torch.Tensor]':
        c4_1, c6_1, sppf_1 = self.backbone1(x1)
        c4_2, c6_2, sppf_2 = self.backbone2(x2)
        c4, c6, sppf = self.fusor(c4_1, c6_1, sppf_1, c4_2, c6_2, sppf_2)
        cls1, ctr1, bbox1, cls2, ctr2, bbox2, cls3, ctr3, bbox3 = self.head(c4, c6, sppf)
        return cls1, ctr1, bbox1, cls2, ctr2, bbox2, cls3, ctr3, bbox3


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


class YOLOLoss():
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
                    # ctr_val = (min(l, r) * min(t, b)) / (max(l, r) * max(t, b))

        return cls, bbox

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, ) -> torch.Tensor:
        # pred: (clases, bboxes)
        # target: (batch, (x0, y0, x1, y1, cls))
        # cls: (batch, num_class, h, w)
        # bbox: (batch, 4, h, w, reg_max)

        batch = pred[0].shape[0]
        channels = pred[0].shape[1]
        image_size = pred[0].shape[2:]
        cls_gt, bbox_gt = self.encode(target, channels, image_size)

        cls_loss = self.focal(pred[0], cls_gt).mean()
        # bbox_loss where cls_gt > 0
        bbox_loss = 0
        for i in range(batch):
            if cls_gt[i].sum() > 0:
                bbox_loss += self.mse(pred[2][i], bbox_gt[i])
        loss = cls_loss + bbox_loss
        return loss


class TinyBackbone(nn.Module):
    """
    Accept two images as input, output a feature map
    
    """

    def __init__(self,) -> None:
        super().__init__()

        self.stride = 8
        self.out_channels = 64

        self.extractor1 = nn.Sequential(
            # 3*640*640 -> 16*320*320
            Conv(3, 16, 3, 2, 1),
            # 16*320*320 -> 64*160*160
            Conv(16, 32, 3, 2, 1),
            # 32*160*160 -> 64*80*80
            Conv(32, 64, 3, 2, 1),
        )
        self.extractor2 = nn.Sequential(
            # 3*640*640 -> 16*320*320
            Conv(3, 16, 3, 2, 1),
            # 16*320*320 -> 64*160*160
            Conv(16, 32, 3, 2, 1),
            # 32*160*160 -> 64*80*80
            Conv(32, 64, 3, 2, 1),
        )

        self.fusor = ConvFusor(64*2, self.out_channels, 3, 1, 1)

    def forward(self, x1, x2) -> 'tuple[torch.Tensor]':
        f1 = self.extractor1(x1)
        f2 = self.extractor2(x2)
        f = self.fusor(f1, f2)

        return f


class TinyHead(nn.Module):
    def __init__(self, channels, num_anchors, num_classes, num_convs) -> None:
        super().__init__()
        self.channels = channels # channels
        self.num_anchors = num_anchors
        self.num_class = num_classes

        convs = [ Conv(channels, channels, 3, 1, 1) for _ in range(num_convs) ]
        self.regressor = nn.Sequential(
            *convs,
            nn.Conv2d(channels, num_anchors*4, 1, 1, 0),
        )
        convs = [ Conv(channels, channels, 3, 1, 1) for _ in range(num_convs) ]
        self.classifier = nn.Sequential(
            *convs,
            nn.Conv2d(channels, num_classes*num_anchors, 1, 1, 0),
        )

    def forward(self, x) -> 'tuple[torch.Tensor]':
        classes = self.classifier(x)
        boxes = self.regressor(x)

        return classes, boxes


class TinyYNet(nn.Module):
    def __init__(self,
        channels=64,
        reg_max=8,
        num_classes=20,
        ) -> None:
        super().__init__()

        self.backbone = TinyBackbone()
        self.head = TinyHead(channels, reg_max, num_classes)


    def forward(self, x1, x2) -> 'tuple[torch.Tensor]':

        feature = self.backbone(x1, x2)
        head_output = self.head(feature)

        
        return head_output
    
    def predict(self, x1, x2) -> 'tuple[torch.Tensor]':
        assert NotImplementedError
        cls, ctr, box = self.forward(x1, x2)
        torchvision.ops.nms(box, ctr, 0.5)
        return cls, box


class FCOS(nn.Module):
    """
    Implements FCOS.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps. For FCOS, only set one anchor for per position of each level, the width and height equal to
            the stride of feature map, and set aspect ratio = 1.0, so the center of anchor is equivalent to the point
            in FCOS paper.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FCOS
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        >>> # FCOS needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280,
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((8,), (16,), (32,), (64,), (128,)),
        >>>     aspect_ratios=((1.0,),)
        >>> )
        >>>
        >>> # put the pieces together inside a FCOS model
        >>> model = FCOS(
        >>>     backbone,
        >>>     num_classes=80,
        >>>     anchor_generator=anchor_generator,
        >>> )
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        # transform parameters
        min_size: int = 640,
        max_size: int = 1920,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # Anchor parameters
        anchor_generator: Optional[AnchorGenerator] = None,
        head: Optional[nn.Module] = None,
        center_sampling_radius: float = 1.5,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.6,
        detections_per_img: int = 2,
        topk_candidates: int = 1000,
        **kwargs,
    ):
        super().__init__()
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        if not isinstance(anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"anchor_generator should be of type AnchorGenerator or None, instead  got {type(anchor_generator)}"
            )

        if anchor_generator is None:
            anchor_sizes = ((8,), )  # equal to strides of multi-level feature map
            aspect_ratios = ((1.0,),) * len(anchor_sizes)  # set only one anchor
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.anchor_generator = anchor_generator
        if self.anchor_generator.num_anchors_per_location()[0] != 1:
            raise ValueError(
                f"anchor_generator.num_anchors_per_location()[0] should be 1 instead of {anchor_generator.num_anchors_per_location()[0]}"
            )

        if head is None:
            head = FCOSHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes,
                # num_convs=2,
                )
        self.head = head

        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        self.center_sampling_radius = center_sampling_radius
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(
        self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses

        return detections

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        num_anchors_per_level: List[int],
    ) -> Dict[str, Tensor]:
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            gt_boxes = targets_per_image["boxes"]
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # Nx2
            anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2  # N
            anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]
            # center sampling: anchor point must be close enough to gt center.
            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(
                dim=2
            ).values < self.center_sampling_radius * anchor_sizes[:, None]
            # compute pairwise distance between N points and M boxes
            x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
            x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
            pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # (N, M)

            # anchor point must be inside gt
            pairwise_match &= pairwise_dist.min(dim=2).values > 0

            # each anchor is only responsible for certain scale range.
            lower_bound = anchor_sizes * 4
            lower_bound[: num_anchors_per_level[0]] = 0
            upper_bound = anchor_sizes * 8
            upper_bound[-num_anchors_per_level[-1] :] = float("inf")
            pairwise_dist = pairwise_dist.max(dim=2).values
            pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])

            # match the GT box with minimum area, if there are multiple GT matches
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # N
            pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
            min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match
            matched_idx[min_values < 1e-5] = -1  # unmatched anchors are assigned -1

            matched_idxs.append(matched_idx)

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(
        self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]], image_shapes: List[Tuple[int, int]]
    ) -> List[Dict[str, Tensor]]:
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        box_ctrness = head_outputs["bbox_ctrness"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            box_ctrness_per_image = [bc[index] for bc in box_ctrness]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, box_ctrness_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, box_ctrness_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sqrt(
                    torch.sigmoid(logits_per_level) * torch.sigmoid(box_ctrness_per_level)
                ).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections

    def forward(
        self,
        images: Tuple[List[Tensor], List[Tensor]],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training:

            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )

        original_image_sizes: List[Tuple[int, int]] = []
        before_images, after_images = images
        for img in after_images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(after_images, targets)
        before_images = self.transform(before_images)[0]

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        f"All bounding boxes should have positive height and width. Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        features = self.backbone([before_images.tensors, images.tensors])
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the fcos heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)
        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                # compute the losses
                losses = self.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)
        else:
            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                print("FCOS always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)


SIZES = {
    'n': (0.33, 0.25, 2.0),
    's': (0.33, 0.50, 2.0),
    'm': (0.50, 0.67, 1.5),
    'l': (1.00, 1.00, 1.0),
    'x': (1.00, 1.25, 1.0),
    }


def check_size():
    net = FCOS(
        backbone=YNetBackbone(*SIZES['s']),
        num_classes=3,
        anchor_generator=AnchorGenerator(
            sizes=((8,), (16,), (32,),),  # equal to strides of multi-level feature map
            aspect_ratios=((1.0,),) * 3, # equal to num_anchors_per_location
        ),
        score_thresh=0.2,
        nms_thresh=1e-5,
        detections_per_img=2,
        topk_candidates=64,
        )

    param_num = 0
    for param in net.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print("Total number of parameters: {} M".format(param_num / 1e6))


if __name__ == '__main__':
    check_size()