"""Compute depth maps for monocular images
"""
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'MiDaS'))

import torch
import numpy as np
import cv2
from tqdm import tqdm
from midas.model_loader import load_model


def normalize(depth):
    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    if depth_max - depth_min > np.finfo("float").eps:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros(depth.shape, dtype=depth.dtype)

    depth_reversed = (1 - depth_normalized)

    return depth_reversed


def run(image_list, model_path, model_type="dpt_beit_large_512", height=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    model, transform, net_w, net_h = load_model(device=device,
                                                model_path=model_path,
                                                model_type=model_type,
                                                optimize=False,
                                                height=height)

    image_prob = cv2.cvtColor(cv2.imread(str(image_list[0])), cv2.COLOR_BGR2RGB) / 255.0
    target_size = image_prob.shape[:-1]
    depth_maps = np.empty((len(image_list),) + target_size, dtype=np.float32)

    for idx, img_file in tqdm(enumerate(image_list), total=len(image_list),
                             desc="Monocular Depth estimation"):

        orig_image = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": orig_image})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(image).to(device).unsqueeze(0)
            prediction = model.forward(sample)
            prediction = (torch.nn.functional.interpolate(prediction.unsqueeze(1),
                                                          size=target_size,
                                                          mode="nearest").squeeze().cpu().numpy())
            prediction = normalize(prediction)
        depth_maps[idx] = prediction

    return depth_maps


def save_depth(depth_maps, image_list, output_path):
    depth_path = output_path / 'depth'
    if not depth_path.exists():
        depth_path.mkdir(parents=True)

    for idx in tqdm(range(len(image_list)), desc="Depth Maps saving"):
        depth_file = str(depth_path / image_list[idx].stem) + '.png'

        # 16-bit encoding (https://developers.google.com/depthmap-metadata/encoding)
        max_value = 2**16-1
        depth_map = max_value*depth_maps[idx]
        cv2.imwrite(depth_file, depth_map.astype(np.uint16))

    print(f'Depth maps saved in: {str(depth_path)}')

