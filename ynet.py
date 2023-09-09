
import torch
from torch import nn
import torchvision
from torchvision.models.resnet import ResNet, ResNet18_Weights
from torchvision.ops import boxes as ops


class Conv(nn.Module):
    # basic Conv block used in YOLOv8
    def __init__(self, c_in, c_out, k=3, s=2, p=1, ):
        super(Conv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Extractor(nn.Module):
    # YOLO-like feature extractor

    def __init__(self, width_mul=0.25):
        super(Extractor, self).__init__()
        self.width_mul = width_mul

        wid = int(64 * self.width_mul)
        self.net = nn.Sequential(
            # 640*640*3 -> 320*320*64*w
            Conv(3, wid, 3, 2, 1),
            # 320*320*64*w -> 160*160*128*w
            Conv(wid, wid*2, 3, 2, 1),
            # 160*160*128*w -> 80*80*256*w
            Conv(wid*2, wid*4, 3, 2, 1),
            # 80*80*256*w -> 40*40*512*w
            Conv(wid*4, wid*8, 3, 2, 1),
            # 40*40*512*w -> 20*20*512*w
            Conv(wid*8, wid*8, 3, 2, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Detector(nn.Module):
    # YOLO-like detector
    # input: two feature maps
    # output: bbox and class prediction
    def __init__(self, width_mul=0.25, num_class=3):
        super(Detector, self).__init__()
        self.width_mul = width_mul

        wid = int(64 * self.width_mul)
        self.combine = Conv(wid*8*2, wid*8, 3, 1, 1)

        # bbox regressor
        self.regressor = nn.Sequential(
            # 20*20*512*w -> 20*20*512*w
            Conv(wid*8, wid*8, 3, 1, 1),
            # 20*20*512*w -> 20*20*512*w
            Conv(wid*8, wid*8, 3, 1, 1),
            nn.Conv2d(wid*8, 4, 1),
        )

        # output class and center
        self.classifier = nn.Sequential(
            # 20*20*512*w -> 20*20*512*w
            Conv(wid*8, wid*8, 3, 1, 1),
            # 20*20*512*w -> 20*20*512*w
            Conv(wid*8, wid*8, 3, 1, 1),
            nn.Conv2d(wid*8, num_class+1, 1),
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.combine(x)
        bbox = self.regressor(x)
        cls = self.classifier(x)

        return torch.cat((bbox, cls), dim=1)


class YNet(nn.Module):
    # YNet consists of two encoders and one detector
    @staticmethod
    def loss_fn(pred: torch.Tensor, target: torch.Tensor,
        class_loss_fn=nn.CrossEntropyLoss, box_loss_fn=ops.generalized_box_iou
        ) -> torch.Tensor:
        # target: batch*(object_num*5+1)
        labels, boxes = target

        # pred: batch*num_grid*num_grid*(num_bboxes*5+num_classes)
        class_loss = 0
        conf_loss = 0
        box_loss = 0
        for i in range(pred.shape[0]):
            # batch
            for j in range(pred.shape[1]):
                for k in range(pred.shape[2]):
                    # num_grid*num_grid, each grid
                    box1 = pred[i, j, k, :4]
                    conf1 = pred[i, j, k, 4]
                    box2 = pred[i, j, k, 5:9]
                    conf2 = pred[i, j, k, 9]
                    class_vec = pred[i, j, k, 10:]

                    label_box = boxes[i]

                    if labels[i] == 0:
                        class_loss += class_loss_fn(pred[i, j, k, 10:], labels[i])
                        conf_loss += torch.norm(pred[i, j, k, 4:9])
    
    def __init__(self, ):
        super(YNet, self).__init__()

        self.extractor1 = Extractor()
        self.extractor2 = Extractor()
        self.detector = Detector()


    def forward(self, x1, x2) -> torch.Tensor:
        x1 = self.extractor1(x1)
        x2 = self.extractor2(x2)
        x = self.detector(x1, x2)
        return x


def test():
    net = YNet()
    x1 = torch.randn(32, 3, 640, 640)
    x2 = torch.randn(32, 3, 640, 640)
    detect_mat = net.forward(x1, x2)
    print(detect_mat.shape)
    param_num = 0
    for param in net.parameters():
        param_num += param.numel()
    print("Total number of parameters: {} M".format(param_num / 1e6))



if __name__ == '__main__':
    test()