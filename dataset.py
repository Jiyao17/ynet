
import os
import copy
import json
from typing import NewType, Tuple

from torch.utils.data import Dataset
import torch
from torchvision import transforms, ops

import numpy as np
from PIL import Image



Data = NewType('Data', 
    # (before_img, after_img), (cls, ctr, bbox)
    Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
    )

TrainingData = NewType('TrainingData',
    # (before_img, after_img), (cls, ctr, bbox)
    Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    )


class DatasetHandler():
    labels = ['nothing', 'airpods', 'cellphone', 'wallet', 'key', 'card', ]

    def __init__(self, filename: str, train_num: int, test_num: int,) -> None:
        super().__init__()
        self.filename = filename
        self.train_num = train_num
        self.test_num = test_num
        self.items = {}

        self.raw_items = self.parse_organize(filename)
        self.paired_train_items, self.paired_test_items = self.split_raw_dataset(
            self.raw_items, train_num, test_num
        )

        
        # store processed data in memory
        self.buffer: 'dict[int, list[torch.Tensor]]' = {}
        self.encoded: 'dict[int, list[torch.Tensor]]' = {}

    @staticmethod
    def parse_organize(filename: str):
        with open(filename, 'r') as f:
            file = json.load(f)
            raw_items = []
            for asset in file['assets']:
                path = asset['asset']['path']
                targets = []
                for region in asset['regions']:
                    label = region['tags'][0]['name']
                    bbox = region['boundingBox']
                    x, y, w, h = bbox['left'], bbox['top'], bbox['width'], bbox['height']
                    label = DatasetHandler.labels.index(label)
                    targets.append((x, y, w, h, label))

                raw_items.append((path, targets))

        return raw_items

    @staticmethod
    def split_raw_dataset(items, train_num: int, test_num: int):
        """
        """
        nothing_items = [ item for item in items if item[1] == [] ]
        other_items = [ item for item in items if item[1] != [] ]

        train_nothing = nothing_items[:train_num]
        test_nothing = nothing_items[train_num:train_num+test_num]
        train_other = other_items[:train_num]
        test_other = other_items[train_num:train_num+test_num]

        # sample nothing images
        train_nothing_items = np.random.choice(train_nothing, train_num, replace=True)
        test_nothing_items = np.random.choice(test_nothing, test_num, replace=True)
        # sample other images
        train_other_items = np.random.choice(train_other, train_num, replace=True)
        test_other_items = np.random.choice(test_other, test_num, replace=True)

        # merge two lists
        paired_train_items = [ 
            (nothing, other) for nothing, other in zip(train_nothing_items, train_other_items)
        ]
        paired_test_items = [
            (nothing, other) for nothing, other in zip(test_nothing_items, test_other_items)
        ]

        return paired_train_items, paired_test_items


class LBIDDataset(Dataset):
    def __init__(self, items, transform=None) -> None:
        super().__init__()
        self.items = items
        self.transform = transform

        # store processed data in memory
        self.buffer: 'dict[int, TrainingData]' = {}

    def __getitem__(self, index: int):
        if index in self.buffer:
            return self.buffer[index]
        
        before, after = self.items[index]
        # read jpg image and convert to tensor
        before_img, before_boxes = before
        after_img, after_boxes = after
        if self.transform is not None:
            before_img = self.transform(before_img)
            after_img = self.transform(after_img)

        after_boxes = torch.tensor(after_boxes, dtype=torch.float32)

        return (before_img, after_img), after_boxes
    
    def __len__(self):
        return len(self.items)



def test():
    IMG_DIR = '/home/jiyao/project/lbod/ynet/raw_ds/'
    BOX_DIR = '/home/jiyao/project/lbod/ynet_od/dataset/train_json/'
    generator = DatasetHandler(IMG_DIR, BOX_DIR)
    ds = generator.ds_gen(
        {'nothing':10000, 'airpods':10000, 'cellphone':10000},
        {'nothing':2000, 'airpods':2000, 'cellphone':2000}
    )

    dataset = LBODDataset(ds)
    # show a random image pair from dataset
    import random
    import matplotlib.pyplot as plt
    index = random.randint(0, len(dataset))
    (before, after), (label, box) = dataset[index]
    
    plt.subplot(1, 2, 1)
    plt.imshow(before)
    plt.subplot(1, 2, 2)
    # add a bounding box to after image
    # box = [x, y, w, h]
    x, y, w, h = (int(i) for i in box)
    line_width = 5
    after = np.array(after)
    after[y:y+line_width, x:x+w, :] = 255
    after[y+h-line_width:y+h, x:x+w, :] = 255
    after[y:y+h, x:x+line_width, :] = 255
    after[y:y+h, x+w-line_width:x+w, :] = 255

    after = Image.fromarray(after)
    
    plt.imshow(after)
    plt.title(dataset.labels[label])
    plt.show()



if __name__ == "__main__":
    test()