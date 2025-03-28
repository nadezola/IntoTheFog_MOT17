from tqdm import tqdm
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import statistics

import opt


def horizon_intensity(image, depth):
    depth_threshold = statistics.mode(depth.flatten()) - opt.sky_threshold
    # inf_pixels = image[depth > opt.sky_threshold * depth.max()]
    inf_pixels = image[depth > depth_threshold]
    atm_light = inf_pixels.mean()
    if atm_light < 0.5:
        atm_light = 0.5
    elif atm_light > 0.7:
        atm_light = 0.7
    return atm_light


def dark_channel(image):
    patch_size = opt.dark_channel_patch
    top_percent = opt.dark_channel_top
    dark_channel = cv2.erode(image, np.ones((patch_size, patch_size), np.uint8))

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # image1 = cv2.cvtColor(dark_channel.astype(np.float32), cv2.COLOR_RGB2GRAY)
    # ax.imshow(image1, cmap='gray')
    # #fig.savefig(out_root / 'turbulence_map.png')
    # plt.show()
    # plt.close()

    numpx = int(image.size * top_percent)
    top_indices = np.argpartition(dark_channel.flatten(), -numpx)[-numpx:]
    atm_light = image.flatten()[top_indices].mean()

    return atm_light

def intensity(image):
    atm_light = image.mean()

    return atm_light

def brightest10(image):
    values = np.sort(image.flatten())
    percent = int(values.shape[0] * 0.1)
    atm_light = (values[-percent:]).mean()

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # image1 = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    # ax.imshow(image1, cmap='gray')
    # #fig.savefig(out_root / 'turbulence_map.png')
    # plt.show()
    # plt.close()

    return atm_light
