

import os


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, ops

from tqdm import tqdm
import numpy as np
from PIL import Image

from dataset import InCarDSODGenerator, LBODDataset
from nn import YNetOD


# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 20
NUM_WORKERS = 7
PIN_MEMORY = False
INPUT_SIZE = 224

LOAD_MODEL = False
IMG_DIR = "/home/jiyao/project/lbod/dataset/raw/images/"
BOX_DIR = "/home/jiyao/project/lbod/dataset/raw/jsons/"
CHECKPOINT_DIR = "/home/jiyao/project/lbod/checkpoint/"

def train_fn(
        dataloader,
        model: YNetOD,
        optimizer: optim.Adam,
        clsf_loss_fn, box_loss_fn,
        scale: torch.Tensor
        ):
    loop = tqdm(dataloader)
    for batch_idx, (data, targets) in enumerate(loop):
        img1, img2 = data[0].to(DEVICE), data[1].to(DEVICE)
        label, box = targets[0].to(DEVICE), targets[1].to(DEVICE)
        scale = scale.to(DEVICE)
        # scale the box
        box[:, 0], box[:, 2] = box[:, 0]*scale[0], box[:, 2]*scale[0]
        box[:, 1], box[:, 3] = box[:, 1]*scale[1], box[:, 3]*scale[1]

        # forward
        pred_label, pred_box = model(img1, img2)

        # classification loss
        clsf_loss: torch.Tensor = clsf_loss_fn(pred_label, label)

        # additional penalty for diverging extractors weights
        extract_loss = 0
        params1, params2 = model.extractor1.named_parameters(), model.extractor2.named_parameters()
        for (name1, param1), (name2, param2) in zip(params1, params2):
            if name1 == name2:
                extract_loss += torch.norm(param1 - param2)

        # box loss
        # transform x, y, w, h to x1, y1, x2, y2
        pred_box = ops.box_convert(pred_box, 'xywh', 'xyxy')
        box = ops.box_convert(box, 'xywh', 'xyxy')
        box_loss: torch.Tensor = box_loss_fn(pred_box, box).diag()
        # plus a penalty of corner distance
        # box_loss += torch.norm(pred_box-box, dim=1)/INPUT_SIZE
        # for nothing images, box loss is 0
        box_loss = torch.where(label == 0, torch.zeros_like(box_loss), box_loss)

        # backward
        optimizer.zero_grad()
        loss = clsf_loss + box_loss.mean()
        loss.backward()
        optimizer.step()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def test_fn(
        dataloader, model: nn.Module,
        clsf_loss_fn, box_loss_fn,
        scale
        ):
    num_correct = 0
    num_samples = 0
    nothing_num = 0
    loss = 0
    recalls = [0, 0, 0]
    box_dice = 0
    model.eval()

    with torch.no_grad():
        loop = tqdm(dataloader)
        for x, y in loop:
            img1, img2 = x[0].to(DEVICE), x[1].to(DEVICE)
            label, box = y[0].to(DEVICE), y[1].to(DEVICE)
            scale = scale.to(DEVICE)
            # scale the box
            box[:, 0], box[:, 2] = box[:, 0]*scale[0], box[:, 2]*scale[0]
            box[:, 1], box[:, 3] = box[:, 1]*scale[1], box[:, 3]*scale[1]


            pred_label, pred_box = model(img1, img2)
            clsf_loss: torch.Tensor = clsf_loss_fn(pred_label, label)
            # transform x, y, w, h to x1, y1, x2, y2
            pred_box = ops.box_convert(pred_box, 'xywh', 'xyxy')
            box = ops.box_convert(box, 'xywh', 'xyxy')
            box_loss: torch.Tensor = box_loss_fn(pred_box, box).diag()
            # for nothing images, box loss is 0
            box_loss = torch.where(label == 0, torch.zeros_like(box_loss), box_loss)
            loss += clsf_loss.item() + box_loss.mean().item()

            num_correct += (pred_label.argmax(1) == label).sum()
            num_samples += pred_label.size(0)
            dices = ops.box_iou(pred_box, box).diag()
            # for nothing images, box dice is 0
            dices = torch.where(label == 0, torch.zeros_like(dices), dices)
            nothing_num += (label == 0).sum()
            box_dice += dices.sum().item()


            loop.set_postfix(
                accu=float(num_correct)/float(num_samples),
                loss=loss/num_samples,
                box_dice=box_dice/(num_samples-nothing_num))

    model.train()
    accu = float(num_correct)/float(num_samples)
    loss = loss/num_samples
    box_dice = box_dice/(num_samples-nothing_num)

    return accu, loss, box_dice

def save_checkpoint(dataloader, model: nn.Module, labels_set, scale, save_dir: str):
    import matplotlib.pyplot as plt
    import os
    model.eval()

    # save predictions as images
    with torch.no_grad():
        for x, y in dataloader:
            img1, img2 = x[0].to(DEVICE), x[1].to(DEVICE)
            label, box = y[0].to(DEVICE), y[1].to(DEVICE)

            pred_label, pred_box = model(img1, img2)
            _, predictions = pred_label.max(1)
            # save predictions as images
            # plot img1 and img2 as subplots, add box to img2, and pred as title
            transform = transforms.ToPILImage()
            pic1 = [transform(img) for img in img1]
            pic2 = [transform(img) for img in img2]
            width_pred = 5
            width_gt = 3
            for i in range(len(pic2)):
                if label[i] == 0:
                    continue

                # draw prediction box on pic2
                pic2[i] = np.array(pic2[i])
                x1, y1, x2, y2 = ops.box_convert(pred_box[i], 'xywh', 'xyxy')
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                pic2[i][y1:y1+width_pred, x1:x2, :] = 255
                pic2[i][y2-width_pred:y2, x1:x2, :] = 255
                pic2[i][y1:y2, x1:x1+width_pred, :] = 255
                pic2[i][y1:y2, x2-width_pred:x2, :] = 255

                # draw ground truth box on pic2
                box[i][0] *= scale[0]
                box[i][1] *= scale[1]
                box[i][2] *= scale[0]
                box[i][3] *= scale[1]
                x1, y1, x2, y2 = ops.box_convert(box[i], 'xywh', 'xyxy')
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                pic2[i][y1:y1+width_gt, x1:x2, :] = 0
                pic2[i][y2-width_gt:y2, x1:x2, :] = 0
                pic2[i][y1:y2, x1:x1+width_gt, :] = 0
                pic2[i][y1:y2, x2-width_gt:x2, :] = 0

                pic2[i] = Image.fromarray(pic2[i])


            pred = predictions.cpu().numpy()
            for i in range(len(pred)):
                if i > 32:
                    break
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(pic1[i])
                ax2.imshow(pic2[i])
                plt.title(labels_set[pred[i]])
                plt.savefig(os.path.join(save_dir, str(i)+'.png'))
                plt.close(fig)

            break

    # save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

    model.train()


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    generator = InCarDSODGenerator(IMG_DIR, BOX_DIR)
    generator.split(400, 100)
    train_ds, test_ds = generator.ds_gen(
        {'nothing':4000, 'airpods':4000, 'cellphone':4000},
        {'nothing':1000, 'airpods':1000, 'cellphone':1000}
        )
    trainset = LBODDataset(train_ds, transform)
    testset = LBODDataset(test_ds, transform)
    image_size = trainset.image_size
    scale = torch.tensor([INPUT_SIZE/image_size[0], INPUT_SIZE/image_size[1]], dtype=torch.float32)
    labels_set = trainset.labels

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    
    model = YNetOD().to(DEVICE)
    print(sum(p.numel() for p in model.parameters()))
    if LOAD_MODEL:
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'model.pth')))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    clsf_loss = F.cross_entropy
    box_loss = ops.complete_box_iou_loss

    for epoch in range(NUM_EPOCHS):
        train_fn(trainloader, model, optimizer, clsf_loss, box_loss, scale)
        print("Testing...")
        accu, loss, box_dice = test_fn(testloader, model, clsf_loss, box_loss, scale)
        print(f"Epoch {epoch}: accuracy = {accu*100}%, loss = {loss}, box_dice = {box_dice}")
        print("Saving predictions...")
        save_checkpoint(testloader, model, labels_set, scale, CHECKPOINT_DIR)

