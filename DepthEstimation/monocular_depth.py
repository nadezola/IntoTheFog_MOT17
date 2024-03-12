import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'MiDaS'))

import torch
import numpy as np
import cv2
from tqdm import tqdm
from midas.model_loader import load_model
from DepthEstimation import utils, temporal_consistency

import opt

def run(image_list, output_path, model_path, model_type="dpt_beit_large_512", height=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    model, transform, net_w, net_h = load_model(device=device,
                                                model_path=model_path,
                                                model_type=model_type,
                                                optimize=False,
                                                height=height)

    t0_inv_depth_map = None
    inv_depth_maps = []
    for idx, img_file in tqdm(enumerate(image_list), total=len(image_list),
                             desc="Monocular Depth estimation"):

        orig_image = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB) / 255.0
        target_size = orig_image.shape[:-1]
        image = transform({"image": orig_image})["image"]
        with torch.no_grad():
            sample = torch.from_numpy(image).to(device).unsqueeze(0)

            # Prediction
            pred_inv_depth_map = model.forward(sample)

            # Interpolate to original size
            pred_inv_depth_map = (torch.nn.functional.interpolate(pred_inv_depth_map.unsqueeze(1),
                                                                  size=target_size,
                                                                  mode="nearest").squeeze().cpu().numpy())
        # Align maps
        if t0_inv_depth_map is not None:
            aligned_inv_depth_map = temporal_consistency.align_depth_maps(pred_inv_depth_map, t0_inv_depth_map)
        else:
            aligned_inv_depth_map = pred_inv_depth_map

        t0_inv_depth_map = aligned_inv_depth_map

        # Normalize map
        norm_inv_depth_map = utils.normalize(aligned_inv_depth_map)

        # Save map
        if opt.save_norm_inv_depth:
            utils.save_norm_depth(norm_inv_depth_map, output_path / 'inv_norm_depth', img_file.stem)

        inv_depth_maps.append(norm_inv_depth_map)

    return np.array(inv_depth_maps)


def load(image_list, depth_path):
    inv_depth_maps = []
    for idx, img_file in tqdm(enumerate(image_list), total=len(image_list),
                             desc="Loading depth maps"):
        depth_file = depth_path / f'{img_file.stem}.png'
        assert depth_file.exists(), f'Depth map {depth_file} does not exist!'

        norm_inv_depth_map = utils.load_norm_depth(depth_file)
        inv_depth_maps.append(norm_inv_depth_map)

    return np.array(inv_depth_maps)
