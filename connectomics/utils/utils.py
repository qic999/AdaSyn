'''
Descripttion: 
version: 0.0
Author: Wei Huang
Date: 2023-02-15 20:59:04
'''
import cv2
import torch
import random
import logging
import numpy as np
import torch.nn.functional as F

def init_logging(path='./log_file.log'):
    logging.basicConfig(
                        filename = path,
                        filemode = 'w',
                        datefmt  = '%m-%d %H:%M',  # '%m-%d-%Y %H:%M:%S'
                        format   = '%(message)s',  # '%(asctime)s: %(message)s'
                        level    = logging.INFO)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def nms(hmap, kernel=21):
    hmax = F.max_pool3d(hmap, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == hmap).float()
    return hmap * keep

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian3D(shape, sigma=1):
    m, n, p = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-m:m + 1, -n:n + 1, -p:p + 1]

    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_umich_gaussian_3D(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian3D((diameter, diameter, diameter), sigma=diameter / 6)

    z, y, x = int(center[0]), int(center[1]), int(center[2])

    depth, height, width = heatmap.shape[0:3]

    front, back = min(z, radius), min(depth - z, radius + 1)
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[z - front:z+back, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - front:radius + back, radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def center_crop(image, det_shape=[160, 160, 160]):
    # To prevent overflow
    # image = np.pad(image, ((20,20),(20,20),(20,20)), mode='reflect')
    src_shape = image.shape
    shift0 = (src_shape[0] - det_shape[0]) // 2
    shift1 = (src_shape[1] - det_shape[1]) // 2
    shift2 = (src_shape[2] - det_shape[2]) // 2
    assert shift0 > 0 or shift1 > 0 or shift2 > 0, "overflow in center-crop"
    image = image[shift0:shift0+det_shape[0], shift1:shift1+det_shape[1], shift2:shift2+det_shape[2]]
    return image

def weight_binary_ratio(label, mask=None, alpha=1.0):
    """Binary-class rebalancing."""
    # input: numpy tensor
    # weight for smaller class is 1, the bigger one is at most 20*alpha
    if label.max() == label.min(): # uniform weights for single-label volume
        weight_factor = 1.0
        weight = np.ones_like(label, np.float32)
    else:
        label = (label!=0).astype(int)
        if mask is None:
            weight_factor = float(label.sum()) / np.prod(label.shape)
        else:
            weight_factor = float((label*mask).sum()) / mask.sum()
        weight_factor = np.clip(weight_factor, a_min=5e-2, a_max=0.99)

        if weight_factor < 0.5:
            weight = label + alpha*weight_factor/(1-weight_factor)*(1-label)
        else:
            weight = alpha*(1-weight_factor)/weight_factor*label + (1-label)

        if mask is not None:
            weight = weight*mask

    return weight.astype(np.float32)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def _adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power):
    lr = lr_poly(learning_rate, i_iter, max_iters, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power)
