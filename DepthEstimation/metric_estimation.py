import cv2
import numpy as np

from DepthEstimation import plot

import opt

# # Correspondence points in the image
# image_pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
#
# # Size of the object in the scene (in real-world units)
# object_size = 10  # for example, in centimeters
#
# # Find homography matrix
# H, _ = cv2.findHomography(image_pts, image_pts)
#
# # Transform image points to new coordinate system
# transformed_pts = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
#
# # Calculate distances in the transformed coordinate system
# distances_pixels = np.sqrt(np.sum(np.diff(transformed_pts, axis=0)**2, axis=1))
#
# # Convert distances from pixels to real-world units
# pixels_per_unit = np.mean(distances_pixels) / object_size
# distances_real_world = distances_pixels / pixels_per_unit
#
# print("Distances between corresponding points in real-world units:", distances_real_world)


def metric(normalized_map, min_dist, max_dist):
    scale = (1 / min_dist) - (1 / max_dist)
    shift = 1 / max_dist
    metric_map = 1 / (scale * normalized_map + shift)

    return metric_map


def run(norm_inv_depth_maps, image_list, output_path):
    metric_depth_maps = []
    for norm_inv_depth, img_path in zip(norm_inv_depth_maps, image_list):
        seq_name = img_path.parent.stem
        metric_depth = metric(norm_inv_depth, min_dist=opt.seq_info[seq_name]['min_dist'],
                                              max_dist=opt.seq_info[seq_name]['max_dist'])
        if opt.plot_metric_depth:
            plot.inverse_and_depth_maps(norm_inv_depth, metric_depth, output_path / 'metric_depth', img_path.stem)

        metric_depth_maps.append(metric_depth)

    return np.array(metric_depth_maps)