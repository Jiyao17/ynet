

import torch
from torch import Tensor
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

import numpy as np
from PIL import Image
import tqdm
import matplotlib.pyplot as plt

from utils.dataset import LBIDRawDataset, LBIDTensorDataset, TensorDataset, collate_fn
from utils.utils import plot_boxes_on_img
from model import FCOS, TinyBackbone, FCOSBackbone


def train(
        model: FCOS, 
        trainloader: DataLoader, 
        optimizer: torch.optim.Optimizer,
        device: str='cuda',
        save_file: str=None,
        ):
    loss = 0
    train_num = 0
    model.train()
    model.to(device)
    loop = tqdm.tqdm(trainloader, total=len(trainloader))
    for i, ((x1, x2), targets) in enumerate(loop):
        x1 = x1.to(device)
        x2 = x2.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # losses = model([x1, x2], targets)
        losses = model(x2, targets)

        optimizer.zero_grad()
        losses: Tensor = sum(losses.values())
        losses.backward()
        optimizer.step()
        
        loss += losses.item()
        train_num += len(x1)

        loop.set_description(f'loss: {loss / train_num}')

    if save_file is not None:
        torch.save(model.state_dict(), save_file)

    return loss / train_num

def test(
    model: FCOS,
    testloader: DataLoader, 
    device: str='cuda',
    ):
    def multi_label_match(labels: Tensor, labels_gt: Tensor):
        labels: list = labels.cpu().numpy().tolist()
        labels_gt: list = labels_gt.cpu().numpy().tolist()

        obj_num = len(labels_gt)
        if labels_gt[0] == 0:
            if labels[0] == 0:
                return 1, 0

        predicted = 0
        recalled = 0

        for label in labels:
            if label in labels_gt:
                predicted += 1
                if label != 0:
                    recalled += 1

                labels_gt.remove(label)

        return total_correct
    
    model.eval()
    model.to(device)
    correct = 0
    total_pred = 0
    total_target = 0
    with torch.no_grad():
        for i, ((x1, x2), targets) in enumerate(testloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # detections = model([x1, x2])
            detections = model(x2)
            for j, detection in enumerate(detections):
                detection = detections[0]
                boxes, labels, scores = detection['boxes'], detection['labels'], detection['scores']
                # idxes = nms(boxes, scores, 0.5)
                # boxes = detection['boxes'][idxes]
                
                # remove background
                # boxes = boxes[labels != 0]
                # labels = labels[labels != 0]
                # scores = scores[labels != 0]

                labels_gt = targets[j]['labels']
                boxes_gt = targets[j]['boxes']

                if len(labels) == 0:
                    labels = torch.tensor([0])

                predicted, recalled = multi_label_match(labels, labels_gt)
                total_pred += len(labels)
                total_target += len(labels_gt)
                # iou = torchvision.ops.box_iou(boxes, boxes_gt)

    return correct / total_pred, correct / total_target
                
def plot_result(
    model: FCOS,
    testloader: DataLoader,
    device: str='cuda',
    save_dir: str=None
    ):
    model.eval()
    model.to(device)
    count = 0
    max_num = 16
    with torch.no_grad():
        for i, ((x1, x2), targets) in enumerate(testloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # detections = model([x1, x2])
            detections = model(x2)


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


class LBID():
    def __init__(self, 
        dataset_dir='/home/jiyao/project/ynet/dataset/raw/',
        train_nums={'nothing': 180, 'other': 800},
        test_nums={'nothing': 45, 'other': 200},
        expand=(2, 2),
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
            expand=expand,
        )
        self.trainloader, self.testloader = self.get_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.labels = LBIDRawDataset.labels

        self.model = FCOS(
            backbone=FCOSBackbone(1, 16, 2),
            num_classes=len(self.labels),
            anchor_generator=AnchorGenerator(
                sizes=((8,), (16,), (32,),),
                aspect_ratios=((0.5,), (1.0,), (2,),), # equal to num_anchors_per_location
            ),
            score_thresh=0.165,
            nms_thresh=0.01,
            detections_per_img=2,
            topk_candidates=64,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_datasets(self, dataset_dir, train_nums, test_nums, expand
        ):
        self.dataset = LBIDRawDataset(dataset_dir)
        trainset, testset = self.dataset.split_raw_dataset(train_nums, test_nums, expand)
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

    def load_model(self, saved_model='./checkpoint/checkpoint.pth'):
        self.model.load_state_dict(torch.load(saved_model))
        self.model.to(self.device)

    def train(self, save_file: str=None):
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

        if save_file is not None:
            torch.save(self.model.state_dict(), save_file)

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

                detections = self.model([x1, x2])
                for j, detection in enumerate(detections):
                    boxes, labels, scores = detection['boxes'], detection['labels'], detection['scores']
                    labels_gt = targets[j]['labels']

                    pred_vec, target_vec, match_vec = labels_to_vec(labels, labels_gt)
                    all_pred_vec += pred_vec
                    all_target_vec += target_vec
                    all_match_vec += match_vec

        nothing_match = all_match_vec[0]
        nothing_pred = all_pred_vec[0]
        items_pred = all_pred_vec[1:]
        items_target = all_target_vec[1:]
        items_match = all_match_vec[1:]
        if nothing_match == 0:
            sensitivity = torch.inf
        else:
            sensitivity = nothing_match / nothing_pred
        # calc mAP, set to 0 if no item detected
        AP = [ torch.tensor(0.0) if items_pred[i] == 0 else items_match[i] / items_pred[i] for i in range(len(items_pred)) ]
        mAP = torch.mean(torch.stack(AP))
        # calc mAR, set to 0 if no item detected
        AR = [ torch.tensor(0.0) if items_target[i] == 0 else items_match[i] / items_target[i] for i in range(len(items_target)) ]
        mAR = torch.mean(torch.stack(AR))

        return sensitivity, mAP, mAR

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





    
if __name__ == '__main__':
    # torch.random.manual_seed(0)
    # np.random.seed(0)
    
    TRAIN_MODE = True
    LOAD_MODEL = False
    saved_model='./checkpoint/checkpoint.pth'

    dataset_dir='/home/jiyao/project/ynet/dataset/raw/'
    # train_nums={'nothing': 180, 'other': 600}
    # test_nums={'nothing': 45, 'other': 200}
    train_nums={'nothing': 180, 'other': 800}
    test_nums={'nothing': 45, 'other': 200}

    expand=(5, 3)

    EPOCH=100
    BATCH_SIZE=16
    LR=0.0001
    NUM_WORKERS=7
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lbid = LBID(
        dataset_dir=dataset_dir,
        train_nums=train_nums,
        test_nums=test_nums,
        expand=expand,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        lr=LR,
        device=DEVICE,
    )

    if TRAIN_MODE:
        if LOAD_MODEL:
            lbid.load_model(saved_model)
        for epoch in range(EPOCH):
            train_loss = lbid.train(save_file=saved_model)
            print(f'epoch: {epoch}, train_loss: {train_loss:.2f}', end=' ')
            
            sensitivity, mAP, mAR = lbid.test()
            print(f'sensitivity: {sensitivity}, mAP: {mAP}, mAR: {mAR}')
            print('model score: {:.2f}'.format(mAP + mAR))
    else:
        assert LOAD_MODEL, 'LOAD_MODEL must be True when TRAIN_MODE is False'
        assert saved_model is not None, 'saved_model must be specified when TRAIN_MODE is False'
        lbid.load_model(saved_model)
        sensitivity, mAP, mAR = lbid.test()
        print('sensitivity: {:.2f}, mAP: {:.2f}, mAR: {:.2f}'.format(sensitivity, mAP, mAR))
        print('model score: {:.2f}'.format(mAP + mAR))

        lbid.plot_result('./checkpoint/')