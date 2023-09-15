

import json
from typing import NewType, Tuple, List

import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

Filename = NewType('Filename', str)
BBox = NewType('BBox', Tuple[int, int, int, int])
Label = NewType('Label', int)

# data entry for LBIDRawDataset
DataEntry = Tuple[Filename, Tuple[BBox, Label]]
# DataEntry = NewType('DataEntry', Tuple[Filename, BBox, Label])
# RawDataset = NewType('RawDataset', List[Tuple[DataEntry, DataEntry]])
RawDataset = List[Tuple[DataEntry, DataEntry]]

# data entry for LBIDDataset
# TensorDataEntry = NewType('TensorDataEntry', Tuple[torch.Tensor, torch.Tensor])
TensorDataEntry = Tuple[torch.Tensor, torch.Tensor]
# TensorDataset = NewType('TensorDataset', List[TensorDataEntry])
TensorDataset = List[TensorDataEntry]



class LBIDRawDataset():
    labels = ['nothing', 'airpod', 'phone', 'wallet', 'key', 'card', ]

    def __init__(self, dir: str,) -> None:
        super().__init__()
        self.dir = dir
        self.image_dir = dir + '/images/'

        self.items: RawDataset = []
        self.raw_items = self.parse_organize()

    def parse_organize(self,) -> RawDataset:
        filename = self.dir + '/labels.json'
        with open(filename, 'r') as f:
            raw_items = json.load(f)
            for item in raw_items:
                filename: str = item['image']
                filename = self.image_dir + filename.split('-')[-1]

                gts = []
                if 'label' in item:
                    raw_labels: list[dict] = item['label']
                    for raw_label in raw_labels:
                        x, y, w, h, _, classes, ow, oh = list(raw_label.values())
                        x, y, w, h = int(x*ow/100), int(y*oh/100), int(w*ow/100), int(h*oh/100)
                        label: Label = LBIDRawDataset.labels.index(classes[0])
                        gts.append(((x, y, w, h), label))

                self.items.append((filename, gts))

        return self.items

    def split_raw_dataset(self, train_num: int, test_num: int) \
        -> 'Tuple[RawDataset, RawDataset]':
        nothing_items = [ item for item in self.items if item[1] == [] ]
        other_items = [ item for item in self.items if item[1] != [] ]

        train_nothing = nothing_items[:train_num]
        test_nothing = nothing_items[train_num:train_num+test_num]
        train_other = other_items[:train_num]
        test_other = other_items[train_num:train_num+test_num]

        # sample nothing images
        train_nothing_idx = np.random.choice(range(len(train_nothing)), train_num, replace=True)
        test_nothing_idx = np.random.choice(range(len(test_nothing)), test_num, replace=True)
        train_nothing_items = [ train_nothing[i] for i in train_nothing_idx ]
        test_nothing_items = [ test_nothing[i] for i in test_nothing_idx ]
        # sample other images
        train_other_idx = np.random.choice(range(len(train_other)), train_num, replace=True)
        test_other_idx = np.random.choice(range(len(test_other)), test_num, replace=True)
        train_other_items = [ train_other[i] for i in train_other_idx ]
        test_other_items = [ test_other[i] for i in test_other_idx ]

        # merge two lists
        paired_train_items: RawDataset = [ 
            (nothing, other) for nothing, other in zip(train_nothing_items, train_other_items)
        ]
        paired_test_items: RawDataset = [
            (nothing, other) for nothing, other in zip(test_nothing_items, test_other_items)
        ]

        return paired_train_items, paired_test_items


class LBIDTensorDataset(Dataset):
    def __init__(self, dataset: RawDataset, transform) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

        # store processed data in memory
        self.items: 'dict[int, TensorDataEntry]' = {}

    def __getitem__(self, index: int):
        if index in self.items:
            return self.items[index]
        
        before, after = self.dataset[index]
        before_file = before[0]
        after_file, after_targets = after
        # read jpg image and convert to tensor
        before_img = Image.open(before_file)
        after_img = Image.open(after_file)
        if self.transform is not None:
            before_img = self.transform(before_file)
            after_img = self.transform(after_file)

        after_boxes = [ target[0] for target in after_targets ]
        after_labels = [ target[1] for target in after_targets ]
        after_boxes = torch.as_tensor(after_boxes, dtype=torch.int64)
        after_labels = torch.as_tensor(after_labels, dtype=torch.int64)

        return (before_img, after_img), (after_boxes, after_labels)
    
    def __len__(self):
        return len(self.dataset)


def test(dataset_dir: str):
    ds = LBIDRawDataset(dataset_dir,)
    trainset, testset = ds.split_raw_dataset(2, 1)
    trainset: TensorDataset = LBIDTensorDataset(trainset, None)
    testset: TensorDataset = LBIDTensorDataset(testset, None)
    # show a random image pair from dataset
    import random
    import matplotlib.pyplot as plt
    index = random.randint(0, len(trainset) - 1)
    (before, after), (boxes, labels) = trainset[index]
    labels = [ LBIDRawDataset.labels[i] for i in labels ]
    
    plt.subplot(1, 2, 1)
    plt.imshow(before)
    plt.subplot(1, 2, 2)
    # add a bounding box to after image
    # box = [x, y, w, h]
    after = np.array(after)
    for box in boxes:
        x, y, w, h = box
        line_width = 5
        after[y:y+line_width, x:x+w, :] = 255
        after[y+h-line_width:y+h, x:x+w, :] = 255
        after[y:y+h, x:x+line_width, :] = 255
        after[y:y+h, x+w-line_width:x+w, :] = 255
    after = Image.fromarray(after)
    
    plt.imshow(after)
    plt.title(labels)
    plt.show()


if __name__ == '__main__':
    dataset_dir = '/home/jiyao/project/ynet/dataset/raw/'
    test(dataset_dir)
