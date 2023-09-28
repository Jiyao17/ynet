
import torch
from torch import nn
from torchvision import ops

from src.models.nn import Conv, C2f, SPPF, DFL
from src.models.yolov8 import YOLOv8, SIZES
from src.utils.tal import make_anchors, dist2bbox, bbox2dist


class DetectionLoss:

    def __init__(self, 
        num_class=20,
        reg_max=8,
        strides=[8, 16, 32],
        offset=0.5):
        super().__init__()

        self.num_class = num_class
        self.reg_max = reg_max
        self.num_outputs = num_class + reg_max*4
        self.strides = strides
        self.offset = offset
        
        self.dfl = DFL(reg_max)
        self.bce = nn.BCEWithLogitsLoss()
        self.ciou = ops.complete_box_iou_loss

    def __call__(self, pred: torch.Tensor, anchors, targets: torch.Tensor) -> torch.Tensor:
        """
        pred: list[torch.Tensor], each element is (batch, num_outputs, h, w)
            from each detector
        target: batch * dict[boxes, labels]
        """
        batch = len(targets)
        # encode targets
        for i, target in enumerate(targets):
            targets[i]['boxes'] = bbox2dist(target['boxes'], self.strides, self.offset)
            targets[i]['labels'] = target['labels'].long()



if __name__ == "__main__":
    img = torch.randn(8, 3, 640, 640)
    target = torch.randn(8, 4)
    model = YOLOv8(SIZES['n'], 20, 4)
    pred = model(img)
    loss_fn = DetectionLoss()
    loss = loss_fn(pred, target)
    print(loss)
    