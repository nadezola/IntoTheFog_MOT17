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
    output_path = output_path / 'plots'
    if not output_path.exists():
        output_path.mkdir(parents=True)

    x = np.arange(depth.shape[1])
    y = np.arange(depth.shape[0])
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=plt.figaspect(0.25))

    # Depth subplot
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_surface(xx, yy, depth, cmap=cm.coolwarm)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Depth Map')

    # Invert depth subplot
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot_surface(xx, yy, inv_depth, cmap=cm.coolwarm)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Inverse Depth Map')

    # Distribution plot
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(inv_depth.flatten())
    ax.set_xlabel('Pixels')
    ax.set_title('Distribution')

    fig.savefig(str(output_path / file_name) + '.png')
    plt.close(fig)


def save_depth(inv_depth, output_path, file_name):
    output_path = output_path / 'depth'
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # 16-bit encoding (https://developers.google.com/depthmap-metadata/encoding)
    max_value = 2**16-1
    inv_depth = max_value * inv_depth
    inv_depth_file = str(output_path / file_name) + '.png'
    cv2.imwrite(inv_depth_file, inv_depth.astype(np.uint16))


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

    return depth_normalized



def run(image_list, output_path, model_path, model_type="dpt_beit_large_512", height=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    model, transform, net_w, net_h = load_model(device=device,
                                                model_path=model_path,
                                                model_type=model_type,
                                                optimize=False,
                                                height=height)

    depth_maps = []
    for idx, img_file in tqdm(enumerate(image_list), total=len(image_list),
                             desc="Monocular Depth estimation"):

        orig_image = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB) / 255.0
        target_size = orig_image.shape[:-1]
        image = transform({"image": orig_image})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(image).to(device).unsqueeze(0)

            # Prediction
            inv_depth = model.forward(sample)

            # Interpolate to original size
            inv_depth = (torch.nn.functional.interpolate(inv_depth.unsqueeze(1),
                                                         size=target_size,
                                                         mode="nearest").squeeze().cpu().numpy())
            # Normalize
            inv_depth_normalize = normalize(inv_depth)
            save_depth(inv_depth_normalize, output_path, img_file.stem)

            # Metric Depth
            scale = (1 / opt.min_dist) - (1 / opt.max_dist)
            shift = 1 / opt.max_dist
            depth = 1/(scale * inv_depth_normalize + shift)
            plot_depth_map(inv_depth_normalize, depth, output_path, img_file.stem)

        depth_maps.append(depth)

    return np.array(depth_maps)

