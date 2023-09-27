
from typing import Union, List, Tuple

import numpy as np

import torch


def matched_preds(
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    labels_gt: Union[np.ndarray, torch.Tensor, List[int]],
    num_classes: int) -> tuple:
    if len(labels) == 0:
        labels = [0]

    pred_vec = np.zeros(num_classes)
    target_vec = np.zeros(num_classes)
    for label in labels:
        pred_vec[round(label)] += 1
    for label in labels_gt:
        target_vec[round(label)] += 1
    match_vec = np.minimum(pred_vec, target_vec)

    return pred_vec, target_vec, match_vec

def stats(pred: np.ndarray, target: np.ndarray, match: np.ndarray) -> tuple:
    AP, AR = match / pred, match / target
    mAP, mAR = AP.mean(), AR.mean()
    ap, ar = match.sum() / pred.sum(), match.sum() / target.sum()

    return AP, AR, mAP, mAR, ap, ar

def plot_boxes_on_img(
        img: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        color=(255, 255, 255),
        width=5
        ) -> np.ndarray:
    """
    img: np.ndarray
    boxes: list of box [x1, y1, x2, y2]
    """

    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img[y1:y1+width, x1:x2, :] = color
        img[y2-width:y2, x1:x2, :] = color
        img[y1:y2, x1:x1+width, :] = color
        img[y1:y2, x2-width:x2, :] = color
        
    return img

