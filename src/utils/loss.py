
import torch
from torch import nn
from torchvision import ops


class DetectionLoss:

    def __init__(self):
        
        self.bce = nn.BCEWithLogitsLoss()
        self.ciou = ops.complete_box_iou_loss

    def __call__(self, pred, target) -> torch.Tensor:
        """
        pred: (batch, num_outputs, num_anchors)
        target: batch * dict[boxes, labels]
        """
        
        pass