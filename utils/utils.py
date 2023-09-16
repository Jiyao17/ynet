

import numpy as np
from PIL import Image


def plot_box_on_img(
        img: np.ndarray,
        boxes: 'list[tuple[int, int, int, int]]',
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


    
    