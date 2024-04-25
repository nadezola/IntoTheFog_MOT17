import torch
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
import logging

from DepthEstimation import plot, utils
import opt

logger = logging.getLogger('FogRendering')


def align_depth_maps(depth_maps):
    aligned_depth_maps = []
    y = depth_maps[0][::10, ::10].reshape(-1, 1)
    y_min = 0
    y_max = 1
    for idx in tqdm(range(len(depth_maps)), desc='Aligning depth maps'):
        depth_map_t1 = depth_maps[idx]
        X = depth_map_t1[::10, ::10].reshape(-1, 1)

        ransac = RANSACRegressor()
        ransac.fit(X, y)
        scale = ransac.estimator_.coef_[0]
        translate = ransac.estimator_.intercept_

        aligned_depth = scale*depth_map_t1 + translate

        if y_min < aligned_depth.min():
            y_min = aligned_depth.min()
        if y_max > aligned_depth.max():
            y_max = aligned_depth.max()
        aligned_depth_maps.append(aligned_depth)

    aligned_depth_maps = np.array(aligned_depth_maps)
    aligned_depth_maps[aligned_depth_maps < y_min] = y_min
    aligned_depth_maps[aligned_depth_maps > y_max] = y_max
    aligned_depth_maps = aligned_depth_maps.clip(0, 1)
    logger.info(f'Aligned depth min = {aligned_depth_maps.min()} :: Aligned depth max = {aligned_depth_maps.max()}')
    return np.array(aligned_depth_maps)

