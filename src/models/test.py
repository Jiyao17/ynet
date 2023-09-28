
import torch

from models.yolov8 import YOLOv8, SIZES
from utils.loss import DetectionLoss

if __name__ == "__main__":
    img = torch.randn(8, 3, 640, 640)
    target = torch.randn(8, 4)
    size = SIZES['n']
    model = YOLOv8(size, 20, 8)
    param_num = sum(p.numel() for p in model.parameters())
    print("param_num:", param_num / 1e6, "M")

    preds = model(img)
    print(preds[0].shape, preds[1].shape, preds[2].shape)

    loss_fn = DetectionLoss()
    loss = loss_fn(preds, target)
    print(loss)

    # yolo = YOLO('../../assets/yolov8n.pt', 'detect')
    # model = yolo.model.DetectionModel()
    # # param_num = sum(p.numel() for p in model.parameters())
    # # print(param_num)
    # summary = torchsummary.summary(model, (3, 224, 224), 1, device='cpu') 
    # print(summary)
    
    # img = torch.randn(1, 3, 640, 640)
    # cls1, bbox1, cls2, bbox2, cls3, bbox3 = yolo(img)
    # print(cls1.shape, bbox1.shape)