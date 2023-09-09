
import os
import copy
import json

from torch.utils.data import Dataset
import torch
from torchvision import transforms, ops

import numpy as np
from PIL import Image



class InCarDSODGenerator():
    """
    This class handles the raw files.
    All data are not read but managed by their paths.
    """
    def __init__(self, image_dir, box_dir) -> None:
        self.image_dir = image_dir
        self.box_dir = box_dir

        # full names of images
        self.images_names = os.listdir(image_dir)
        self.images_labels = [name.split("_")[0] for name in self.images_names]
        self.images_paths = [os.path.join(image_dir, name) for name in self.images_names]

        self.boxes_names = copy.deepcopy(self.images_names)
        self.boxes_names = [name.replace('.jpg', '.json') for name in self.boxes_names]
        self.boxes_paths = [os.path.join(box_dir, name) for name in self.boxes_names]
        
        self.classified_images: 'dict[str, list[tuple[str,str]]]' = {}
        for image_path, box_path, label in zip(self.images_paths, self.boxes_paths, self.images_labels):
            if label not in self.classified_images:
                self.classified_images[label] = []
            self.classified_images[label].append((image_path, box_path))

    def split(self, train_num, test_num):
        self.train_img = {}
        self.test_img = {}

        for label, img_list in self.classified_images.items():
            np.random.shuffle(img_list)
            train_list = img_list[:train_num]
            test_list = img_list[train_num:train_num+test_num]
            self.train_img[label] = train_list
            self.test_img[label] = test_list

        return self.train_img, self.test_img
    
    def ds_gen(self, train_nums: 'dict[str, int]', test_nums: 'dict[str, int]'):
        self.train_ds = {image_label: [] for image_label in self.images_labels}
        self.test_ds = {image_label: [] for image_label in self.images_labels}
        for image_label in train_nums.keys():
            train_range, num = len(self.train_img[image_label]), train_nums[image_label]
            train_befores_indices = np.random.choice(train_range, num, replace=True)
            train_afters_indices = np.random.choice(train_range, num, replace=True)
            test_range, num = len(self.test_img[image_label]), test_nums[image_label]
            test_befores_indices = np.random.choice(test_range, num, replace=True)
            test_afters_indices = np.random.choice(test_range, num, replace=True)

            train_befores = [self.train_img['nothing'][i] for i in train_befores_indices]
            train_afters = [self.train_img[image_label][i] for i in train_afters_indices]
            test_befores = [self.test_img['nothing'][i] for i in test_befores_indices]
            test_afters = [self.test_img[image_label][i] for i in test_afters_indices]

            for (before_img, before_box), (after_img, after_box) in zip(train_befores, train_afters):
                if image_label == 'nothing':
                    after_box = None
                self.train_ds[image_label].append((before_img, after_img, after_box))

            for (before_img, before_box), (after_img, after_box) in zip(test_befores, test_afters):
                if image_label == 'nothing':
                    after_box = None
                self.test_ds[image_label].append((before_img, after_img, after_box))
                
        return self.train_ds, self.test_ds

    def export(self, target_dir: str):
        pass


class LBODDataset(Dataset):
    """
    This class is used to read data from InCarDSODGenerator.
    It reads all data into memory and process them before training,
    i.e., all pre-processing is done before calling __getitem__.
    """
    labels = ['nothing', 'airpods', 'cellphone']

    def __init__(self, 
            ds: 'dict[str, list[tuple]]',
            transform=None,) -> None:
        super().__init__()
        self.ds = ds
        self.transform = transform

        self.items = []
        for label, data_list in ds.items():
            for data in data_list:
                before_path, after_path, box_path = data
                self.items.append(((before_path, after_path), (label, box_path)))

        self.image_size = Image.open(self.items[0][0][0]).size

    def get_rect_mask(self, points: 'tuple[tuple[float]]'):
        """
        accept 4 points
        return a rectangle mask in format of (x1, y1, x2, y2)
        """
        assert len(points) == 4
        x1, y1, x2, y2, x3, y3, x4, y4 = points
        x_min, x_max = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
        y_min, y_max = min(y1, y2, y3, y4), max(y1, y2, y3, y4)

        return x_min, y_min, x_max, y_max

    def read_data(self,):
        self.processed_data = []
        for item in self.items:
            (before_path, after_path), (image_label, box_path) = item
            # read jpg image and convert to tensor
            before_img = Image.open(before_path)
            after_img = Image.open(after_path)
            if self.transform is not None:
                before_img = self.transform(before_img)
                after_img = self.transform(after_img)

            boxes = []
            if box_path is not None:
                with open(box_path, 'r') as f:
                    file = json.load(f)[0]
                    points = file['content']
                    for i in range(len(points)//4):
                        box = self.get_rect_mask(points[i*4:i*4+4])
                        boxes.append(box)
            else:
                box = [0, 0, self.image_size[0], self.image_size[1]]
            label = self.labels.index(image_label)
            # label_vec = torch.zeros(len(self.labels), dtype=torch.float32)
            # label_vec[label] = 1
            box = torch.tensor(box, dtype=torch.float32)

        return (before_img, after_img), (label, box)

    @staticmethod
    def convert_label(target: torch.Tensor, grid_num=7, bbox_num=2, class_num=3):
        # convert label to one-hot vector
        # target: batch*label_size*(label, bbox)
        # return: batch*num_grid*num_grid*(num_bboxes*5+num_classes)
        rtarget = torch.zeros(target.shape[0], grid_num, grid_num, bbox_num*5+class_num)
        for i in range(rtarget.shape[0]):
            # each image
            for j in range(rtarget.shape[1]):
                # each box
                label = target[i, j, 0]
                bbox = target[i, j, 1:]
                bbox = ops.box_convert(bbox, 'xywh', 'cxcywh')

                bbox_idx_x = int(bbox[0] * grid_num)
                bbox_idx_y = int(bbox[1] * grid_num)
                
                rtarget[i, bbox_idx_x, bbox_idx_y, 0] = 1

    def __getitem__(self, index: int):
        (before_path, after_path), (image_label, box_path) = self.items[index]

        # read jpg image and convert to tensor
        before_img = Image.open(before_path)
        after_img = Image.open(after_path)
        if self.transform is not None:
            before_img = self.transform(before_img)
            after_img = self.transform(after_img)

        box = None
        if box_path is not None:
            with open(box_path, 'r') as f:
                file = json.load(f)[0]
                mask = file['rectMask']
                x, y, w, h = mask['xMin'], mask['yMin'], mask['width'], mask['height']
                box = [x, y, w, h]
        else:
            box = [0, 0, self.image_size[0], self.image_size[1]]
        label = self.labels.index(image_label)
        # label_vec = torch.zeros(len(self.labels), dtype=torch.float32)
        # label_vec[label] = 1
        box = torch.tensor(box, dtype=torch.float32)

        return (before_img, after_img), (label, box)
    
    def __len__(self):
        return len(self.items)
    

def test():
    IMG_DIR = '/home/jiyao/project/lbod/ynet/raw_ds/'
    BOX_DIR = '/home/jiyao/project/lbod/ynet_od/dataset/train_json/'
    generator = InCarDSODGenerator(IMG_DIR, BOX_DIR)
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