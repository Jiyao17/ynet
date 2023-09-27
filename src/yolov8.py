
import os

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

from ultralytics import YOLO

from src.utils.hresult import matched_preds, plot_boxes_on_img



def read_label(filename):
    file = open(filename, "r")
    labels = file.read().split("\n")
    boxes = []
    for label in labels:
        if label == "":
            continue
        label = label.split(" ")
        cls = label[0]
        x = float(label[1])
        y = float(label[2])
        w = float(label[3])
        h = float(label[4])
        boxes.append([cls, x, y, w, h])
    return np.array(boxes)



# model = YOLO("./runs/detect/lbid8/weights/best.pt", "detect")
model = YOLO("yolov8.yaml", "detect")
results = model.train(data="dataset/yolo_dataset/lbid.yaml", 
    epochs=30, batch=16, imgsz=640,
    name="lbid"
    )

folder = "/home/jiyao/project/ynet/dataset/yolo_dataset/"
img_dir = os.path.join(folder, "images/test/")
images = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
label_dir = os.path.join(folder, "labels/test/")
filenames = [os.path.join(label_dir, img.replace(".jpg", ".txt")) for img in os.listdir(img_dir)]

preds = model.predict(images)


pred_vec = np.zeros(3)
target_vec = np.zeros(3)
match_vec = np.zeros(3)
for i, pred in enumerate(preds):
    boxes = read_label(filenames[i])
    classes = boxes[:, 0]
    pred_boxes = pred.boxes.data.cpu().numpy()
    pred_classes = pred_boxes[:, -1]
    pred_bboxes = pred_boxes[:, :-2]
    pred, target, match = matched_preds(pred_classes, classes, 3)

    # img = Image.open(images[i])
    # img = np.array(img)
    # plot_img = plot_boxes_on_img(img, pred_bboxes)
    # plt.title(f"pred: {pred}, target: {target}")
    # plt.imshow(plot_img)
    # plt.savefig(f"checkpoint/{i}.png")

    pred_vec += pred
    target_vec += target
    match_vec += match

precision = np.sum(match_vec) / np.sum(pred_vec)
recall = np.sum(match_vec) / np.sum(target_vec)

print("pred_vec: ", pred_vec)
print("target_vec: ", target_vec)
print("match_vec: ", match_vec)
print("precision: ", precision, "recall: ", recall)