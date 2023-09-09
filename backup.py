
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