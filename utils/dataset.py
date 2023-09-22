

import json
import os
from typing import NewType, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision import ops

import numpy as np
from PIL import Image

Filename = NewType('Filename', str)
BBox = NewType('BBox', Tuple[int, int, int, int])
Label = NewType('Label', int)

# data entry for LBIDRawDataset
DataEntry = Tuple[Filename, Tuple[BBox, Label]]
# DataEntry = NewType('DataEntry', Tuple[Filename, BBox, Label])
# RawDataset = NewType('RawDataset', List[Tuple[DataEntry, DataEntry]])
RawDataset = Tuple[List[DataEntry], List[DataEntry]]
# data entry for LBIDDataset
# TensorDataEntry = NewType('TensorDataEntry', Tuple[torch.Tensor, torch.Tensor])
TensorDataEntry = Tuple[torch.Tensor, torch.Tensor]
# TensorDataset = NewType('TensorDataset', List[TensorDataEntry])
TensorDataset = List[TensorDataEntry]



class LBIDRawDataset():
    labels = ['nothing', 'airpod', 'phone', 'bag']

    def __init__(self, ds_dir: str,) -> None:
        super().__init__()
        self.dir = ds_dir
        self.image_dir = ds_dir + '/images/'

        self.stats = np.zeros(len(LBIDRawDataset.labels), dtype=np.int32)

        self.train_items: RawDataset = ([], [])
        self.test_items: RawDataset = ([], [])
        self.items: RawDataset = self.parse_organize()


    def parse_organize(self,) -> RawDataset:
        # parse lebeled data
        filename = self.dir + '/labels.json'
        nothing_items: List[DataEntry] = []
        other_items: List[DataEntry] = []
        with open(filename, 'r') as f:
            raw_items = json.load(f)
            for item in raw_items:
                filename: str = item['image']
                santinized = filename.split('-')[-1]
                filename = self.image_dir + santinized

                gts = []
                raw_labels: list[dict] = item['label']
                for raw_label in raw_labels:
                    x, y, w, h, _, classes, ow, oh = list(raw_label.values())
                    x, y, w, h = int(x*ow/100), int(y*oh/100), int(w*ow/100), int(h*oh/100)
                    x1, y1, x2, y2 = x, y, x+w, y+h

                    label = LBIDRawDataset.labels.index(classes[0])
                    gts.append(((x1, y1, x2, y2), label))
                    self.stats[label] += 1

                other_items.append((filename, gts))


        # parse nothing data
        imgs = os.listdir(self.image_dir)
        imgs = [ self.image_dir + img for img in imgs if img.startswith('nothing') ]
        # danger: 1920, 1080 is hard coded
        nothing_items = [ (img, [((0, 0, 1920, 1080), 0)]) for img in imgs ]
        self.stats[0] += len(nothing_items)

        print("stat:", self.stats)
        self.items = (nothing_items, other_items)
        return self.items

    def split_raw_dataset(self,
        train={'nothing': 4, 'other': 8},
        test={'nothing': 2, 'other': 4},
        shuffle=True) \
        -> 'Tuple[RawDataset, RawDataset]':
        if shuffle:
            np.random.shuffle(self.items[0])
            np.random.shuffle(self.items[1])
        nothing_items, other_items = self.items

        # print("nothing num:", len(nothing_items))
        # print("other num:", len(other_items))

        # split nothing and other items
        train_nothing = nothing_items[:train['nothing']]
        test_nothing = nothing_items[train['nothing']:train['nothing']+test['nothing']]
        train_other = other_items[:train['other']]
        test_other = other_items[train['other']:train['other']+test['other']]

        self.train_items = (train_nothing, train_other)
        self.test_items = (test_nothing, test_other)

        return self.train_items, self.test_items



class LBIDTensorDataset(Dataset):
    def __init__(self, dataset: RawDataset, transform=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

        self.nothing, self.other = dataset
        self.merged_dataset = self.nothing + self.other

    def __getitem__(self, index: int):
        after = self.merged_dataset[index]
        rand_nothing = np.random.randint(0, len(self.nothing))
        before = self.nothing[rand_nothing]
        after_file, after_targets = after
        # read jpg image and convert to tensor
        before_img = Image.open(before[0])
        after_img = Image.open(after_file)
        if self.transform is not None:
            before_img = self.transform(before_img)
            after_img = self.transform(after_img)

        after_boxes = [ target[0] for target in after_targets ]
        after_labels = [ target[1] for target in after_targets ]
        after_boxes = torch.as_tensor(after_boxes, dtype=torch.int64)
        after_labels = torch.as_tensor(after_labels, dtype=torch.int64)
        
        return (before_img, after_img), {'boxes': after_boxes, 'labels': after_labels}
    
    def __len__(self):
        return len(self.merged_dataset)


def collate_fn(batch):
    """
    batch: list of tuple (before_img, after_img), targets
    """
    images = [ (before, after) for (before, after), targets in batch ]
    targets = [ targets for (before, after), targets in batch ]

    before_images = [ before for before, after in images ]
    after_images = [ after for before, after in images ]
    before_images = torch.stack(before_images)
    after_images = torch.stack(after_images)

    return (before_images, after_images), targets


def test(dataset_dir: str):
    ds = LBIDRawDataset(dataset_dir,)
    trainset, testset = ds.split_raw_dataset((2, 1), (2, 2))
    trainset: TensorDataset = LBIDTensorDataset(trainset, None)
    testset: TensorDataset = LBIDTensorDataset(testset, None)
    # show a random image pair from dataset
    import random
    import matplotlib.pyplot as plt
    index = random.randint(0, len(trainset) - 1)
    (before, after), targets = trainset[index]
    boxes, labels = targets['boxes'], targets['labels']
    labels = [ LBIDRawDataset.labels[i] for i in labels ]
    
    plt.subplot(1, 2, 1)
    plt.imshow(before)
    plt.subplot(1, 2, 2)
    # add a bounding box to after image
    # box = [x, y, w, h]
    after = np.array(after)
    for box in boxes:
        x1, y1, x2, y2 = box
        line_width = 5
        after[y1:y1+line_width, x1:x2, :] = 255
        after[y2-line_width:y2, x1:x2, :] = 255
        after[y1:y2, x1:x1+line_width, :] = 255
        after[y1:y2, x2-line_width:x2, :] = 255
    after = Image.fromarray(after)
    
    plt.imshow(after)
    plt.title(labels)
    plt.show()


if __name__ == '__main__':
    dataset_dir = '/home/jiyao/project/ynet/dataset/raw/'
    test(dataset_dir)
