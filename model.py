
import torch
from torch import nn
from torchvision import ops
from torchvision import models

# YNet is a model that takes two images as input,
# and outputs difference of objects in the images.
# 
# YNet consists of 3 parts: two Extractors and one Comparator.
# Extractor is a CNN that takes an image as input,
# and outputs a feature vector.
# Comparator is a MLP that takes two feature vectors as input,
# and outputs difference of the two object sets in the images.
#
# YNet uses two Extractors to extract features from two images,
# and uses Comparator to compare the two feature vectors.
#
# Currently, it is designed only to detect what is added to the second image.

class Extractor(models.ResNet):
    def __init__(self,
        base_block=models.resnet.BasicBlock,
        layers=[2, 2, 2, 2],
        fv_size=1000,
        ):
        super().__init__(base_block, layers)
        if fv_size != self.fc.out_features:
            self.fc = nn.Linear(self.fc.in_features, fv_size)


class Comparator(ops.MLP):
    def __init__(self, layers=[2000, 400, 30, 3],):
        super().__init__(layers[0], layers[1:])

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = super().forward(x)
        return x


class Detector(ops.MLP):
    def __init__(self, layers=[2000, 400, 30, 4],):
        super().__init__(layers[0], layers[1:])

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = super().forward(x)
        x = torch.sigmoid(x)
        return x


class XNet(nn.Module):
    def __init__(self,
        extractor1=Extractor(),
        extractor2=Extractor(),
        comparator=Comparator(),
        detector=Detector(),
        ):
        super().__init__()
        self.extractor1 = extractor1
        self.extractor2 = extractor2
        self.comparator = comparator
        self.detector = detector

    def forward(self, x1, x2):
        x1 = self.extractor1(x1)
        x2 = self.extractor2(x2)
        cls = self.comparator(x1, x2)
        box = self.detector(x1, x2)

        return cls, box


class Regressor(nn.Module):
    # a YoLo-like regressor that takes a feature vector as input,
    # and outputs class and bounding box.

    # accepts a feature vector of size 1024,
    # and outputs a vector of size 4 + 1 + 1 = 6.
    # 4 for bounding box, 1 for class, 1 for confidence.
    def __init__(self, fv_size=1024,):
        super().__init__()
        