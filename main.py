

import torch
from torch import Tensor
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

import numpy as np
from PIL import Image
import tqdm

from utils.dataset import LBIDRawDataset, LBIDTensorDataset, TensorDataset, collate_fn
from utils.utils import plot_box_on_img
from model import FCOS, TinyBackbone


def test():
    transform = Compose([
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
                img = plot_box_on_img(img, boxes)
                img = Image.fromarray(img.astype(np.uint8))
                img.show()


def train(
        model: FCOS, 
        trainloader: DataLoader, 
        optimizer: torch.optim.Optimizer,
        save_file: str=None,
        ):
    loss = 0
    train_num = 0
    model.train()
    for i, ((x1, x2), targets) in enumerate(trainloader):  
        losses = model([x1, x2], targets)

        optimizer.zero_grad()
        losses: Tensor = sum(losses.values())
        losses.backward()
        optimizer.step()
        
        loss += losses.item()
        train_num += len(x1)

    if save_file is not None:
        torch.save(model.state_dict(), save_file)

    return loss / train_num


def test(model: FCOS, testloader: DataLoader, save_dir: str=None):
    model.eval()
    with torch.no_grad():
        for i, ((x1, x2), targets) in enumerate(testloader):
            detections = model([x1, x2])
            for j, detection in enumerate(detections):
                detection = detections[0]
                boxes, labels, scores = detection['boxes'], detection['labels'], detection['scores']
                # idxes = nms(boxes, scores, 0.5)
                # boxes = detection['boxes'][idxes]

                img = x2[j].permute(1, 2, 0).numpy()
                img *= 255
                img = plot_box_on_img(img, boxes)
                img = Image.fromarray(img.astype(np.uint8))
                
                # only save one for each batch
                labels = [ LBIDRawDataset.labels[k] for k in labels ]
                if save_dir is not None:
                    img.save(save_dir + f'/batch{i}-img{j}-{labels}.jpg')

def main(
        dataset_dir='/home/jiyao/project/ynet/dataset/raw/',
        train_nums={'nothing': 4, 'other': 8},
        test_nums={'nothing': 2, 'other': 4},
        expand=(5, 2),
        EPOCH=20,
        BATCH_SIZE=8,
        LR=0.001,
        NUM_WORKERS=4,
        saved_model='./checkpoint/checkpoint.pth',
        ):
    transform = Compose([
        Resize((640, 640)),
        ToTensor(),
    ])
    dataset = LBIDRawDataset(dataset_dir)
    trainset, testset = dataset.split_raw_dataset(train_nums, test_nums, expand)
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
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    trainloader = DataLoader(trainset, BATCH_SIZE, True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn)
    testloader = DataLoader(testset, BATCH_SIZE, True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn)

    if saved_model is not None:
        net.load_state_dict(torch.load(saved_model))

    loop = tqdm.tqdm(range(EPOCH))
    for epoch in loop:
        train_loss = train(net, trainloader, optimizer, './checkpoint/checkpoint.pth')
        loop.set_description(f'Epoch {epoch} Train Loss: {train_loss}')
        test(net, testloader, './checkpoint/')




    
if __name__ == '__main__':
    main(
        dataset_dir='/home/jiyao/project/ynet/dataset/raw/',
        train_nums={'nothing': 4, 'other': 8},
        test_nums={'nothing': 2, 'other': 4},
        expand=(5, 2),
        EPOCH=20,
        BATCH_SIZE=32,
        LR=0.001,
        saved_model='./checkpoint/checkpoint.pth',
        # saved_model=None,
    )
