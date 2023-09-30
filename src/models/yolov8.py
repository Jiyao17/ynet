
from typing import Any, Tuple, List

import torch
from torch import nn
from torchvision import ops

from ultralytics.utils.ops import non_max_suppression as nms
from ultralytics.utils.tal import make_anchors, dist2bbox
from ultralytics.nn.modules import DFL

from src.models.nn import Backbone, Head


class YOLOv8(nn.Module):
    def __init__(self, size: Tuple[float], num_classes :int=20, reg_max: int=4):
        super().__init__()
        self.size = size
        self.num_class = num_classes
        self.reg_max = reg_max
        self.num_outputs = num_classes + reg_max*4
        self.strides = [8, 16, 32]

        depth, width, ratio = size
        self.backbone = Backbone(depth, width, ratio)
        self.head = Head(depth, width, ratio, num_classes, reg_max)
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        self.anchors_points = None
        self.maps_shapes = None

    def forward(self, x) -> Tuple[torch.Tensor]:
        """
        return: if training, return raw outputs from detectors
                    shape = (batch, num_outputs, h, w)
                if inference, return (batch, num_outputs, num_anchors)
        """
        assert self.training, "Please call predict() for inference."
        
        c4, c6, sppf = self.backbone(x)
        output = self.head(c4, c6, sppf)
        return output
        

    def predict(self, x, conf_th, iou_th):
        """
        return shape = (batch, 4 (box) + 1 (conf) + 1 (cls))
        """
        assert not self.training, "Please call forward() for training."

        # same as forward()
        c4, c6, sppf = self.backbone(x)
        output = self.head(c4, c6, sppf)
        
        # generate one shape for each grid cell
        self.maps_shapes = [x.shape for x in output]
        # generate one anchor for each grid cell
        self.anchors_points, self.anchor_strides = \
            (x.transpose(0, 1) for x in make_anchors(output, self.strides, 0.5))
                 
        batch = self.maps_shapes[0][0]
        assert output[0].shape[0] == batch
        # x_cat = torch.cat([x.view(batch, self.num_outputs, -1) for x in output], dim=2)
        out = [x.view(batch, self.num_outputs, -1) for x in output]
        x_cat = torch.cat(out, dim=2)
        assert x_cat.shape[-1] == 8400
        # reg_max*4 for bbox, num_class for cls
        box, cls = x_cat.split([self.reg_max*4, self.num_class], dim=1)
        # one dbox (ltrb) for each anchor
        dbox = self.dfl(box)
        dbox = dist2bbox(self.dfl(box), self.anchors_points.unsqueeze(0), True, 1) * self.anchor_strides
        # sigmoid for cls
        cls = cls.sigmoid()
        # class+box for each anchor
        # shape = (batch, num_outputs, num_anchors)
        y = torch.cat((dbox, cls), dim=1)
        # reshape to (batch, num_anchors, num_outputs)
        # y = y.transpose(1, 2).contiguous()

        # shape = (batch, 4 (box) + 1 (conf) + 1 (cls))
        results = nms(y, conf_th, iou_th, classes=range(self.num_class))
        return results


    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return self.forward(*args, **kwds)
    



SIZES = {
    'n': (0.33, 0.25, 2.0),
    's': (0.33, 0.50, 2.0),
    'm': (0.50, 0.67, 1.5),
    'l': (1.00, 1.00, 1.0),
    'x': (1.00, 1.25, 1.0),
}








