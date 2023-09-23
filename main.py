

import torch
from torch import Tensor
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import detection

import numpy as np
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style

from utils.dataset import LBIDRawDataset, LBIDTensorDataset, TensorDataset, collate_fn
from utils.utils import plot_boxes_on_img
from model import FCOS, YNetBackbone, FCOSBackbone


class YNetTask():
    def __init__(self, 
        dataset_dir='/home/jiyao/project/ynet/dataset/raw/',
        train_nums={'nothing': 180, 'other': 800},
        test_nums={'nothing': 45, 'other': 200},
        batch_size=16,
        num_workers=7,
        lr=0.0001,
        device='cuda',
        ) -> None:
        super().__init__()

        self.device = device

        self.transform = Compose([
        Resize((640, 640)),
        ToTensor(),
        ])

        self.dataset: LBIDRawDataset = None
        self.trainset, self.testset = self.get_datasets(
            dataset_dir=dataset_dir,
            train_nums=train_nums,
            test_nums=test_nums,
        )
        self.trainloader, self.testloader = self.get_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.labels = LBIDRawDataset.labels

        self.model = FCOS(
            backbone=YNetBackbone(0.67, 0.75, 1.5),
            num_classes=len(self.labels),
            anchor_generator=AnchorGenerator(
                sizes=((8,), (16,), (32,),),  # equal to strides of multi-level feature map
                aspect_ratios=((1.0,),) * 3, # equal to num_anchors_per_location
            ),
            score_thresh=2,
            nms_thresh=1e-5,
            detections_per_img=2,
            topk_candidates=64,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_datasets(self, dataset_dir, train_nums, test_nums
        ):
        self.dataset = LBIDRawDataset(dataset_dir)
        trainset, testset = self.dataset.split_raw_dataset(train_nums, test_nums)
        self.trainset: TensorDataset = LBIDTensorDataset(trainset, self.transform)
        self.testset: TensorDataset = LBIDTensorDataset(testset, self.transform)

        return self.trainset, self.testset

    def get_loaders(self,
        batch_size=16,
        num_workers=7,
        ):
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            )
        testloader = DataLoader(
            self.testset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers, 
            collate_fn=collate_fn,
        )

        return trainloader, testloader

    def load_model(self, filename='./checkpoint/checkpoint.pth'):
        self.model.load_state_dict(torch.load(filename))
        self.model.to(self.device)

    def save_model(self, filename='./checkpoint/checkpoint.pth'):
        torch.save(self.model.state_dict(), filename)

    def train(self):
        loss = 0
        train_num = 0
        self.model.train()
        self.model.to(self.device)
        loop = tqdm.tqdm(self.trainloader, total=len(self.trainloader))
        for i, ((x1, x2), targets) in enumerate(loop):
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            losses = self.model([x1, x2], targets)
            # losses = self.model(x2, targets)

            self.optimizer.zero_grad()
            losses: Tensor = sum(losses.values())
            losses.backward()
            self.optimizer.step()
            
            loss += losses.item()
            train_num += len(x1)

            loop.set_description(f'loss: {loss / train_num}')

        return loss / train_num

    def test(self):
        def labels_to_vec(labels: Tensor, labels_gt: Tensor):
            labels: list = labels.cpu().numpy().tolist()
            if len(labels) == 0:
                labels = [0]
            labels_gt: list = labels_gt.cpu().numpy().tolist()

            pred_vec = torch.zeros(len(self.labels))
            target_vec = torch.zeros(len(self.labels))
            for label in labels:
                pred_vec[label] += 1
            for label in labels_gt:
                target_vec[label] += 1
            match_vec = torch.min(pred_vec, target_vec)

            return pred_vec, target_vec, match_vec

        self.model.eval()
        self.model.to(self.device)
        pred_vec = torch.zeros(len(self.labels))
        target_vec = torch.zeros(len(self.labels))
        match_vec = torch.zeros(len(self.labels))
        with torch.no_grad():
            for i, ((x1, x2), targets) in enumerate(self.testloader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                detections = self.model([x1, x2])
                for j, detection in enumerate(detections):
                    boxes, labels, scores = detection['boxes'], detection['labels'], detection['scores']
                    labels_gt = targets[j]['labels']

                    pred, target, match = labels_to_vec(labels, labels_gt)
                    pred_vec += pred
                    target_vec += target
                    match_vec += match

        return pred_vec, target_vec, match_vec

    def plot_result(self, save_dir: str=None):
        self.model.eval()
        self.model.to(self.device)
        count = 0
        max_num = 32
        with torch.no_grad():
            for i, ((x1, x2), targets) in enumerate(self.testloader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                detections = self.model([x1, x2])
                detection = detections[0]
                boxes, labels, scores = detection['boxes'], detection['labels'], detection['scores']
                img0 = x1[0].permute(1, 2, 0).cpu().numpy()
                img = x2[0].permute(1, 2, 0).cpu().numpy()
                img *= 255
                img = plot_boxes_on_img(img, boxes)
                img = Image.fromarray(img.astype(np.uint8))
                
                if save_dir is not None:
                    plt.subplot(1, 2, 1)
                    plt.imshow(img0)
                    plt.subplot(1, 2, 2)
                    plt.imshow(img)
                    labels = [ LBIDRawDataset.labels[i] for i in labels ]
                    if len(labels) == 0:
                        labels = ['nothing']
                    if len(labels) == 2:
                        if labels[0] == 'nothing' and labels[1] == 'nothing':
                            labels = ['nothing']
                    labels_gt = [ LBIDRawDataset.labels[i] for i in targets[0]['labels'] ]

                    plt.title(f'pred: {labels}, gt: {labels_gt}')
                    plt.savefig(save_dir + str(count) + '.png')

                    count += 1
                    if count >= max_num:
                        return


class FCOSTask(YNetTask):
    def __init__(self,
        dataset_dir='/home/jiyao/project/ynet/dataset/raw/',
        train_nums={ 'nothing': 180,'other': 800 },
        test_nums={ 'nothing': 45,'other': 200 }, 
        batch_size=16, 
        num_workers=7, 
        lr=0.0001,
        device='cuda') -> None:
        super().__init__(dataset_dir, train_nums, test_nums,
            batch_size, num_workers, lr, device)
        
        self.model = detection.FCOS(
            backbone=FCOSBackbone(2, 32, 2),
            num_classes=len(self.labels),
            anchor_generator=AnchorGenerator(
                sizes=((8,), (16,), (32,),),
                aspect_ratios=((0.5,), (1.0,), (2,),), # equal to num_anchors_per_location
            ),
            score_thresh=0.2,
            nms_thresh=1e-5,
            detections_per_img=2,
            topk_candidates=64,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        loss = 0
        train_num = 0
        self.model.train()
        self.model.to(self.device)
        loop = tqdm.tqdm(self.trainloader, total=len(self.trainloader))
        for i, ((x1, x2), targets) in enumerate(loop):
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # losses = self.model([x1, x2], targets)
            losses = self.model(x2, targets)

            self.optimizer.zero_grad()
            losses: Tensor = sum(losses.values())
            losses.backward()
            self.optimizer.step()
            
            loss += losses.item()
            train_num += len(x1)

            loop.set_description(f'loss: {loss / train_num}')

        return loss / train_num

    def test(self):
        def labels_to_vec(labels: Tensor, labels_gt: Tensor):
            labels: list = labels.cpu().numpy().tolist()
            if len(labels) == 0:
                labels = [0]
            labels_gt: list = labels_gt.cpu().numpy().tolist()

            pred_vec = torch.zeros(len(self.labels))
            target_vec = torch.zeros(len(self.labels))
            for label in labels:
                pred_vec[label] += 1
            for label in labels_gt:
                target_vec[label] += 1
            match_vec = torch.min(pred_vec, target_vec)

            return pred_vec, target_vec, match_vec

        self.model.eval()
        self.model.to(self.device)
        all_pred_vec = torch.zeros(len(self.labels))
        all_target_vec = torch.zeros(len(self.labels))
        all_match_vec = torch.zeros(len(self.labels))
        with torch.no_grad():
            for i, ((x1, x2), targets) in enumerate(self.testloader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # detections = self.model([x1, x2])
                detections = self.model(x2)
                for j, detection in enumerate(detections):
                    boxes, labels, scores = detection['boxes'], detection['labels'], detection['scores']
                    labels_gt = targets[j]['labels']
                    # print(labels)

                    pred_vec, target_vec, match_vec = labels_to_vec(labels, labels_gt)
                    all_pred_vec += pred_vec
                    all_target_vec += target_vec
                    all_match_vec += match_vec

        return all_pred_vec, all_target_vec, all_match_vec

    def plot_result(self, save_dir: str=None):
        self.model.eval()
        self.model.to(self.device)
        count = 0
        max_num = 32
        with torch.no_grad():
            for i, ((x1, x2), targets) in enumerate(self.testloader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # detections = self.model([x1, x2])
                detections = self.model(x2)
                detection = detections[0]
                boxes, labels, scores = detection['boxes'], detection['labels'], detection['scores']
                img0 = x1[0].permute(1, 2, 0).cpu().numpy()
                img = x2[0].permute(1, 2, 0).cpu().numpy()
                img *= 255
                img = plot_boxes_on_img(img, boxes)
                img = Image.fromarray(img.astype(np.uint8))
                
                if save_dir is not None:
                    plt.subplot(1, 2, 1)
                    plt.imshow(img0)
                    plt.subplot(1, 2, 2)
                    plt.imshow(img)
                    labels = [ LBIDRawDataset.labels[i] for i in labels ]
                    if len(labels) == 0:
                        labels = ['nothing']
                    if len(labels) == 2:
                        if labels[0] == 'nothing' and labels[1] == 'nothing':
                            labels = ['nothing']
                    labels_gt = [ LBIDRawDataset.labels[i] for i in targets[0]['labels'] ]

                    plt.title(f'pred: {labels}, gt: {labels_gt}')
                    plt.savefig(save_dir + str(count) + '.png')

                    count += 1
                    if count >= max_num:
                        return


def stats(pred: Tensor, target: Tensor, match: Tensor):
    AP, AR = match / pred, match / target
    mAP, mAR = torch.mean(AP), torch.mean(AR)
    ap, ar = torch.sum(match) / torch.sum(pred), torch.sum(match) / torch.sum(target)

    # convert to basic type
    AP = AP.cpu().numpy().tolist()
    AR = AR.cpu().numpy().tolist()
    mAP = mAP.cpu().numpy().tolist()
    mAR = mAR.cpu().numpy().tolist()
    ap = ap.cpu().numpy().tolist()
    ar = ar.cpu().numpy().tolist()

    return AP, AR, mAP, mAR, ap, ar



    
if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)
    
    TRAIN_MODE = True
    LOAD_MODEL = False
    saved_model='./checkpoint/checkpoint9.pth'
    # saved_model='./trained/double_backbone1.3.pth'

    dataset_dir='./dataset/raw/'
    # train_nums={'nothing': 40, 'other': 40}
    # test_nums={'nothing': 10, 'other': 10}
    train_nums={'nothing': 600, 'other': 800}
    test_nums={'nothing': 0, 'other': 200}

    EPOCH=100
    if TRAIN_MODE:
        BATCH_SIZE=24
    else:
        BATCH_SIZE=16
    LR=0.0001
    NUM_WORKERS=16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lbid = YNetTask(
        dataset_dir=dataset_dir,
        train_nums=train_nums,
        test_nums=test_nums,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        lr=LR,
        device=DEVICE,
    )

    if TRAIN_MODE:
        if LOAD_MODEL:
            lbid.load_model(saved_model)

        for epoch in range(EPOCH):
            train_loss = lbid.train()
            print(f'epoch: {epoch}, train_loss: {train_loss:.5f}', end=' ')
            
            pred, target, match = lbid.test()
            
            AP, AR, mAP, mAR, ap, ar = stats(pred, target, match)
            print(f'AP: {AP}, AR: {AR}')
            print(f'mAP: {mAP}, mAR: {mAR}, ap: {ap}, ar: {ar}')
            print(Fore.GREEN + f'model score: {mAP + mAR}, {ap + ar}' + Style.RESET_ALL)

            lbid.save_model(f'./checkpoint/checkpoint{epoch}.pth')

    else:
        assert LOAD_MODEL, 'LOAD_MODEL must be True when TRAIN_MODE is False'
        assert saved_model is not None, 'saved_model must be specified when TRAIN_MODE is False'
        
        lbid.load_model(saved_model)
        pred, target, match = lbid.test()
        print(pred)
        print(target)
        print(match)

        AP, AR, mAP, mAR, ap, ar = stats(pred, target, match)
        print(f'AP: {AP}, AR: {AR}')
        print(f'mAP: {mAP}, mAR: {mAR}, ap: {ap}, ar: {ar}')
        print(f'model score: {mAP + mAR}, {ap + ar}')

        lbid.plot_result('./checkpoint/')