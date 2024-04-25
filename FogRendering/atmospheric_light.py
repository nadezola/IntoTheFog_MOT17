from tqdm import tqdm
import cv2
import numpy as np
import math

import opt


def horizon_intensity(image, depth):
    inf_pixels = image[depth > opt.sky_threshold * depth.max()]
    atm_light = inf_pixels.mean()

    return atm_light


def dark_channel(image):
    patch_size = opt.dark_channel_patch
    top_percent = opt.dark_channel_top
    dark_channel = cv2.erode(image, np.ones((patch_size, patch_size), np.uint8))
    numpx = int(image.size * top_percent)
    top_indices = np.argpartition(dark_channel.flatten(), -numpx)[-numpx:]
    atm_light = image.flatten()[top_indices].mean()

    return atm_light
