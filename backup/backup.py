
class Classifier(nn.Module):
    # accepts two feature vectors
    # outputs the difference between the two
    # which is a prediction of new object class
    # 1 out of 3 classes
    def __init__(self, feature_size=512, class_size=3):
        super(Classifier, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        self.net = nn.Sequential(
            nn.Linear(feature_size*2, feature_size//16),   
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_size//16, class_size),
            nn.Sigmoid()
        )
        
    def forward(self, f1, f2):
        x = torch.cat((f1, f2), dim=1)
        x = self.net(x)
        return x


class BBoxRegressor(nn.Module):
    # accepts two feature vectors and a bounding box
    # outputs a bounding box
    def __init__(self, feature_size=512, image_size=(224, 224)):
        super(BBoxRegressor, self).__init__()
        self.feature_size = feature_size
        self.image_size = image_size

        self.net = nn.Sequential(
            nn.Linear(feature_size*2, feature_size//16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_size//16, 4),
            nn.Sigmoid()
        )

    def forward(self, f1, f2):
        x = torch.cat((f1, f2), dim=1)
        x = self.net(x)
        return x


class YNetOD(nn.Module):
    # input image shape is identical to the input shape of the ResNet
    # plus a bounding box
    def __init__(self, feature_size=1024, class_size=3, image_size=(448, 448)):
        super(YNetOD, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.image_size = image_size

        # extractors for feature extraction
        self.extractor1 = Extractor()
        self.extractor2 = Extractor()
        # boxer for object detection
        # self.boxer = nn.Sequential(
        #     nn.Linear(4, 4),
        #     nn.Sigmoid(),
        # )

        # comparator for class prediction
        self.classifier = Classifier(self.feature_size, class_size)
        # detector for object detection
        self.detector = BBoxRegressor(self.feature_size, image_size)


    def forward(self, x1, x2) ->'tuple[torch.Tensor, torch.Tensor]':
        f1 = self.extractor1(x1)
        f2 = self.extractor2(x2)

        label = self.classifier(f1, f2)
        box = self.detector(f1, f2) * self.image_size[0]

        return label, box
    


class Extractor(nn.Module):
    # YOLO-like feature extractor

    def __init__(self,):
        super(Extractor, self).__init__()
        
        self.net = nn.Sequential(
            Conv(3, 16, 3, 2, 1),
            Conv(16, 32, 3, 2, 1),
            Conv(32, 64, 3, 2, 1),
        )


class Detector(nn.Module):
    # YOLO-like detector
    # input: two feature maps
    # output: bbox and class prediction
    def __init__(self, width_mul=0.25, num_class=3):
        super(Detector, self).__init__()
        self.width_mul = width_mul

        wid = int(64 * self.width_mul)
        self.combine = Conv(wid*8*2, wid*8, 3, 1, 1)

        # bbox regressor
        self.regressor = nn.Sequential(
            # 20*20*512*w -> 20*20*512*w
            Conv(wid*8, wid*8, 3, 1, 1),
            # 20*20*512*w -> 20*20*512*w
            Conv(wid*8, wid*8, 3, 1, 1),
            nn.Conv2d(wid*8, 4, 1),
        )

        # output class and center
        self.classifier = nn.Sequential(
            # 20*20*512*w -> 20*20*512*w
            Conv(wid*8, wid*8, 3, 1, 1),
            # 20*20*512*w -> 20*20*512*w
            Conv(wid*8, wid*8, 3, 1, 1),
            nn.Conv2d(wid*8, num_class+1, 1),
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.combine(x)
        cls = self.classifier(x)[:,:-1,:,:]
        center = self.classifier(x)[:,-1,:,:]
        bbox = self.regressor(x)

        return cls, center, bbox

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


    def read_sample(self,):
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

    x1 = torch.randn(4, 3, 640, 640)
    x2 = torch.randn(4, 3, 640, 640)
    targets = [
        {
            "boxes": torch.tensor([
                [16, 16, 32, 32],
                [16, 16, 32, 32],
                [16, 16, 32, 32],
                [16, 16, 32, 32],
                ], dtype=torch.float32),
            "labels": torch.tensor([1, 0, 0, 0], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[1, 1, 2, 2]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[1, 1, 2, 2]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([
                [16, 16, 32, 32],
                [16, 16, 32, 32],
                ], dtype=torch.float32),
            "labels": torch.tensor([2, 1], dtype=torch.int64),
        },
    ]




