

import os
from typing import NewType, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import ops

import numpy as np
from PIL import Image

from src.utils.hresult import plot_boxes_on_img

Filename = NewType('Filename', str)
BBox = NewType('BBox', Tuple[int, int, int, int, int])
# data entry for LBIDRawDataset
DataEntry = Tuple[Filename, Tuple[BBox]]




class LBIDRawDataset():
    labels = None

    def __init__(self,
        image_dir: str,
        label_dir: str,
        img_size: Tuple[int, int],
        labels: List[str],
        ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.labels = labels

        self.stats = np.zeros(len(self.labels)+1, dtype=np.int32)
        self.items: List[DataEntry] = self.parse_items()

    def parse_items(self,) -> List[DataEntry]:
        image_files: List[str] = os.listdir(self.image_dir)
        items: List[DataEntry] = []
        W, H = self.img_size
        for image_file in image_files:
            image_path = self.image_dir + image_file
            label_path = self.label_dir + image_file.replace(".jpg", ".txt")
            with open(label_path, 'r') as f:
                lines = f.readlines()
                lines = [ line.strip() for line in lines ]
                bboxes = []
                for line in lines:
                    label, x, y, w, h = line.split(' ')
                    # convert str to numbers
                    label, x, y, w, h = int(label), float(x), float(y), float(w), float(h)
                    # convert percentage to pixel index
                    # x, y, w, h = int(x*W), int(y*H), int(w*W), int(h*H)
                    # x1, y1, x2, y2 = x-w//2, y-h//2, x+w//2, y+h//2
                    bboxes.append((label, x, y, w, h))
                    # bboxes.append((label, x1, y1, x2, y2))
                    self.stats[label] += 1
                items.append((image_path, bboxes))

        return items


class YoloTensorDataset(Dataset):
    def __init__(self, dataset: LBIDRawDataset, transform=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

        self.items = dataset.items

    def __getitem__(self, index: int):
        item = self.items[index]
        img = Image.open(item[0])
        original_size = np.array(img.size, dtype=np.float32)
        scaled_size = original_size
        if self.transform is not None:
            img = self.transform(img)
            scaled_size = np.array(img.shape[1:], dtype=np.float32)
        # scales = scaled_size / original_size

        boxes = np.array(item[1], dtype=np.float32)
        # boxes[:, 1:] *= scales[0], scales[1], scales[0], scales[1]
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        return img, boxes
    
    def __len__(self):
        return len(self.items)


def collate_fn(batch):
    """
    batch: list of (image, target) tuples
    return: {'batch_idx': batch_idx, 'cls': cls, 'bboxes': bboxes}
    """
    imgs = []
    batch_idx, cls, bboxes = [], [], []
    for i, (img, boxes) in enumerate(batch):
        imgs.append(img)

        batch_idx.append(torch.full((len(boxes), 1), i, dtype=torch.int))
        cls.append(boxes[:, 0])
        bboxes.append(boxes[:, 1:])

    batch_idx = torch.cat(batch_idx, 0)
    cls = torch.cat(cls, 0)
    bboxes = torch.cat(bboxes, 0)

    imgs = torch.stack([img for img, _ in batch], 0)
    gt = {'batch_idx': batch_idx, 'cls': cls, 'bboxes': bboxes}
    return imgs, gt


def test():
    dataset_dir = './dataset/'
    with open(dataset_dir + 'classes.txt', 'r') as f:
        lines = f.readlines()
        labels = [ line.strip() for line in lines ]
    
    raw_dataset = LBIDRawDataset(
        image_dir='./dataset/images/train/',
        label_dir='./dataset/labels/train/',
        img_size=(1920, 1080),
        labels=labels,
    )

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    dataset = YoloTensorDataset(raw_dataset, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    for i, (img, gt) in enumerate(dataloader):
        # show box on image
        img = img[0].permute(1, 2, 0).numpy()
        boxes = gt['bboxes'][0]
        xyxy = ops.box_convert(boxes, 'xywh', 'xyxy').numpy()
        boxes = [ xyxy, ]
        img = plot_boxes_on_img(img, boxes)
        
        # show image
        plt.imshow(img)
        plt.savefig(f"./test_{i}.png")
        break

        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test()
