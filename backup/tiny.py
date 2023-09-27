

from typing import Dict, List, Optional, Tuple, Union, OrderedDict

import torch
from torch import nn, Tensor
import torchvision
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.fcos import FCOSHead
from torchvision.ops import boxes as box_ops, nms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils.model import Conv, ConvFusion
from utils.dataset import LBIDRawDataset, LBIDTensorDataset, TensorDataset, collate_fn
from utils.utils import plot_boxes_on_img


def test():
    transform = torchvision.transforms.Compose([
        Resize((640, 640)),
        ToTensor(),
    ])
    transform = ToTensor()
    dataset = LBIDRawDataset('/home/jiyao/project/ynet/dataset/raw/')
    trainset, testset = dataset.split_raw_dataset((2, 1), (5, 2))
    trainset: TensorDataset = LBIDTensorDataset(trainset, transform)
    testset: TensorDataset = LBIDTensorDataset(testset, transform)
    net = FCOS(
        backbone=TinyBackbone(),
        num_classes=20,
        anchor_generator=AnchorGenerator(
            sizes=((8,), ),  # equal to strides of multi-level feature map
            aspect_ratios=((1.0,),) * 1,  # set only one anchor
        ),
        score_thresh=0.01,
        nms_thresh=0.1,
        detections_per_img=2,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for epoch in range(20):
        for i, ((x1, x2), targets) in enumerate(trainloader):
            
            losses = net([x1, x2], targets)

            optimizer.zero_grad()
            loss: Tensor = sum(losses.values())
            loss.backward()
            optimizer.step()

            print(loss.item())

    net.eval()
    with torch.no_grad():
        for i, ((x1, x2), targets) in enumerate(testloader):
            detections = net([x1, x2])
            for i, detection in enumerate(detections):
                detection = detections[0]
                boxes, labels, scores = detection['boxes'], detection['labels'], detection['scores']
                # idxes = nms(boxes, scores, 0.5)
                # boxes = detection['boxes'][idxes]
                box_num = len(boxes)

                img = x2[i].permute(1, 2, 0).numpy()
                img *= 255
                img = plot_boxes_on_img(img, boxes)
                img = Image.fromarray(img.astype(np.uint8))
                img.show()
    
    

if __name__ == '__main__':
    test()