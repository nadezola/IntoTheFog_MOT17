import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'MiDaS'))

import torch
from tqdm import tqdm
import logging
from midas.model_loader import load_model
from DepthEstimation import utils
import opt

logger = logging.getLogger('FogRendering')


def estimate(cleardata, model_path, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    model, transform, net_w, net_h = load_model(device=device,
                                                model_path=model_path,
                                                model_type=model_type,
                                                optimize=False,
                                                height=None)

    pred_depth_maps = []
    for im_id, orig_image in tqdm(cleardata, desc="Monocular depth estimation :"):
        im = transform({"image": orig_image})["image"]
        with torch.no_grad():
            im = torch.from_numpy(im).to(device).unsqueeze(0)
            pred_depth = model.forward(im)
            pred_depth = (torch.nn.functional.interpolate(pred_depth.unsqueeze(1), size=orig_image.shape[:-1],
                                                          mode="nearest").squeeze().cpu().numpy())
        pred_depth = utils.normalize(pred_depth)
        pred_depth_maps.append(pred_depth)

        if opt.save_depth_gray:
            utils.save_norm_depth(pred_depth, cleardata.depth_root, im_id)
        if opt.save_depth_color:
            utils.save_norm_depth(pred_depth, cleardata.depth_cl_root, im_id, incolor=True)

    return pred_depth_maps


def load(cleardata, depth_path):
    depth_maps = []
    for im_id, _ in tqdm(cleardata, desc="Loading depth maps :"):
        depth_file = depth_path / f'{im_id}.png'
        assert depth_file.exists(), f'Depth map {depth_file} does not exist!'

        depth = utils.load_depth(depth_file)
        depth_maps.append(depth)

    return depth_maps
