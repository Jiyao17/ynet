
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import ops

import tqdm
import matplotlib.pyplot as plt
import numpy as np


from src.models.yolov8 import YOLOv8, SIZES
from src.models.ynet import YNet
from src.utils.hresult import plot_boxes_on_img, matched_preds, stats
from src.dataset import LBIDRawDataset, YoloTensorDataset, YNetTensorDataset
from src.utils.loss import v8DetectionLoss



def train_model(
        dataloader: DataLoader, 
        model: YNet, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: v8DetectionLoss,
        DEVICE='cuda',
        folder='checkpoint/'
        ):
    model.train()
    model.to(DEVICE)
    loop = tqdm.tqdm(dataloader)
    for i, ((backgrounds, images), gt) in enumerate(loop):
        backgrounds = backgrounds.to(DEVICE)
        images = images.to(DEVICE)
        gt['bboxes'] = gt['bboxes'].to(DEVICE)
        gt['cls'] = gt['cls'].to(DEVICE)
        gt['batch_idx'] = gt['batch_idx'].to(DEVICE)

        pred = model((backgrounds, images))
        loss, loss_items = loss_fn(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=sum(loss_items))

    return loss_items

def test_model(dataloader: DataLoader, model: YOLOv8, DEVICE, folder=None):

    model.eval()
    model.to(DEVICE)
    max_img = 10
    pred_vec = np.zeros(model.num_class)
    target_vec = np.zeros(model.num_class)
    match_vec = np.zeros(model.num_class)
    for i, ((backgrounds, images), gt) in enumerate(dataloader):
        backgrounds = backgrounds.to(DEVICE)
        images = images.to(DEVICE)
        pred = model.predict((backgrounds, images), conf_th=0.25, iou_th=0.45)
        # statistics
        for j, img in enumerate(images):
            pd_classes = pred[j][:, -1].detach().cpu()
            mask = (gt['batch_idx'] == j).squeeze()
            cls = gt['cls'].squeeze()
            labels = cls[mask]
            assert len(labels.shape) == 1
            pred_, target_, match_ = matched_preds(pd_classes, labels, model.num_class)
            pred_vec += pred_
            target_vec += target_
            match_vec += match_

        if i < max_img and folder is not None:
            # show sample results (first image in the batch)
            image = images[0].permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            boxes = pred[0][:, :4].detach().cpu().numpy()
            classes = pred[0][:, -1].detach().cpu()

            # boxes = ops.box_convert(boxes, 'xywh', 'xyxy')
            image = plot_boxes_on_img(image, boxes)
            # print("obj num:", len(boxes))
            plt.title(f"pred: {classes.tolist()}")
            plt.imshow(image)
            plt.savefig(folder + f"test_{i}.png")
            plt.close()

    return pred_vec, target_vec, match_vec

def test_ynet():
    dataset_dir = './dataset/'
    num_classes, reg_max = 3, 4
    size = SIZES['n']
    TRAIN_MODE = True
    # TRAIN_MODE = False
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    LRS = [1e-2]*5 + [1e-3]*5 + [1e-4]*5 + [1e-5]*5 + [1e-6]*5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PRETRAINED_MODEL = None
    # PRETRAINED_MODEL = './checkpoint/checkpoint24.pth'
    SAMPLE_OUTPUT = None
    # SAMPLE_OUTPUT = './checkpoint/'

    with open(dataset_dir + 'classes.txt', 'r') as f:
        lines = f.readlines()
        labels = [ line.strip() for line in lines ]
    
    trainset = LBIDRawDataset(
        image_dir='./dataset/images/train/',
        label_dir='./dataset/labels/train/',
        img_size=(1920, 1080),
        labels=labels,
    )
    trainset.categorize()
    cat_filter = {}
    for label in labels:
        cat_filter[label] = 150
    cat_filter['background'] = 500
    trainset.filter(cat_filter)

    testset = LBIDRawDataset(
        image_dir='./dataset/images/test/',
        label_dir='./dataset/labels/test/',
        img_size=(1920, 1080),
        labels=labels,
    )
    testset.categorize()
    cat_filter = {}
    for label in labels:
        cat_filter[label] = 30
    cat_filter['background'] = 0
    testset.filter(cat_filter)
    testset.add_background('./dataset/images/nothing_1_frame373.jpg')

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    trainset = YNetTensorDataset(trainset, transform=transform)
    trainloader = DataLoader(trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=YNetTensorDataset.collate_fn,
        num_workers=NUM_WORKERS,
        )
    testset = YNetTensorDataset(testset, transform=transform)
    testloader = DataLoader(testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=YNetTensorDataset.collate_fn,
        num_workers=NUM_WORKERS,
        )

    model = YNet(size, num_classes, reg_max)
    optimizer = torch.optim.Adam(model.parameters(), lr=LRS[0])
    # bacause loss_fn uses model.device
    model.to(DEVICE)
    loss_fn = v8DetectionLoss(model)
    loss_fn.hyp = [1, 1, 0.1]
    param_num = sum(p.numel() for p in model.parameters())
    print("param_num:", param_num / 1e6, "M")

    # load checkpoint
    if PRETRAINED_MODEL is not None:
        model.load_state_dict(torch.load(PRETRAINED_MODEL))

    for epoch in range(25):
        print(f"epoch: {epoch}")
        if TRAIN_MODE:
            # disable output
            SAMPLE_OUTPUT = None
            optimizer = torch.optim.Adam(model.parameters(), lr=LRS[epoch])
            loss_items = train_model(trainloader, model, optimizer, loss_fn, DEVICE)
            torch.save(model.state_dict(), f"checkpoint/checkpoint{epoch}.pth")
            print(loss_items)

        pred, target, match = test_model(testloader, model, DEVICE, folder=SAMPLE_OUTPUT)
        AP, AR, mAP, mAR, ap, ar = stats(pred, target, match)
        print(f"AP: {AP:}, AR: {AR:}")
        print(f"mAP: {mAP:.4f}, mAR: {mAR:.4f}, ap: {ap:.4f}, ar: {ar:.4f}")



if __name__ == "__main__":
    test_ynet()