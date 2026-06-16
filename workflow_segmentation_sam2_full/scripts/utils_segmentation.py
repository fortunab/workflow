import numpy as np
import torch

def get_bounding_box(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0.0, 0.0, 10.0, 10.0]
    return [
        float(np.min(x_indices)),
        float(np.min(y_indices)),
        float(np.max(x_indices)),
        float(np.max(y_indices))
    ]

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2.0 * intersection + smooth) / (union + smooth)).mean()

def calculate_metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = intersection / union if union > 0 else 1.0
    dice = 2 * intersection / (pred.sum() + target.sum()) if (pred.sum() + target.sum()) > 0 else 1.0
    return iou, dice
