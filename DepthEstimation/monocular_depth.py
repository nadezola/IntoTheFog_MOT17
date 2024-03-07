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
import matplotlib.pyplot as plt
from matplotlib import cm
import opt


def plot_depth_map(inv_depth, depth, output_path, file_name):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    x = np.arange(depth.shape[1])
    y = np.arange(depth.shape[0])
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Depth subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(-xx, yy, -depth+30,  vmin=10., vmax=35., cmap=cm.coolwarm)
    #ax.plot_surface(-xx, yy, depth,  cmap=cm.coolwarm)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Metric Depth')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    ax.azim = -45
    ax.elev = 45

    # Invert depth subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(-xx, yy, inv_depth, cmap=cm.coolwarm)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Aligned normalized MiDaS inverse depth')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    ax.azim = -45
    ax.elev = 45

    fig.savefig(str(output_path / file_name) + '.png')
    plt.close(fig)


def save_depth(inv_depth, output_path, file_name):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # 16-bit encoding (https://developers.google.com/depthmap-metadata/encoding)
    max_value = 2**16-1
    inv_depth = max_value * inv_depth
    inv_depth_file = str(output_path / file_name) + '.png'
    cv2.imwrite(inv_depth_file, inv_depth.astype(np.uint16))


def normalize(values, scope=(0, 1)):
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite values present")

    values_min = values.min()
    values_max = values.max()

    if values_max - values_min > np.finfo("float").eps:
        scale = (scope[1] - scope[0]) / (values_max - values_min)
        shift = scope[0] - scale * values_min
        values_normalized = scale * values + shift
    else:
        values_normalized = np.zeros(values.shape, dtype=values.dtype)

    return values_normalized


def align_depth(map, ref_map):
    grid_step = 100
    map_grid = map[::grid_step, ::grid_step].flatten()
    ref_grid = ref_map[::grid_step, ::grid_step].flatten()

    A = np.vstack([map_grid, np.ones(len(map_grid))]).T
    s, t = np.linalg.lstsq(A, ref_grid, rcond=None)[0]

    aligned_map = s * map + t

    return aligned_map


def metric(normalized_map, min_dist, max_dist):
    scale = (1 / min_dist) - (1 / max_dist)
    shift = 1 / max_dist
    metric_map = 1 / (scale * normalized_map + shift)

    return metric_map


def run(image_list, output_path, model_path, model_type="dpt_beit_large_512", height=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    model, transform, net_w, net_h = load_model(device=device,
                                                model_path=model_path,
                                                model_type=model_type,
                                                optimize=False,
                                                height=height)

    t0_inv_depth_map = None
    depth_maps = []
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
        # Align depth
        if t0_inv_depth_map is not None:
            aligned_inv_depth_map = align_depth(pred_inv_depth_map, t0_inv_depth_map)
        else:
            aligned_inv_depth_map = pred_inv_depth_map

        t0_inv_depth_map = aligned_inv_depth_map

        # Normalize inv_depth
        norm_inv_depth_map = normalize(aligned_inv_depth_map)

        # Metric Depth
        seq_name = img_file.parent.stem
        depth_map = metric(norm_inv_depth_map,
                           min_dist=opt.metric_info[seq_name]['min_dist'],
                           max_dist=opt.metric_info[seq_name]['max_dist'])

        # Save and visualize
        save_depth(norm_inv_depth_map, output_path / 'inv_relative_depth', img_file.stem)
        plot_depth_map(norm_inv_depth_map, depth_map, output_path / 'plots', img_file.stem)

        depth_maps.append(depth_map)

    return np.array(depth_maps)

