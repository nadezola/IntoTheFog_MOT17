import numpy as np
from tqdm import tqdm
from DepthEstimation import plot
import opt


def metric(norm_inv_map, min_dist, max_dist):
    s = (1 / min_dist) - (1 / max_dist)
    t = 1 / max_dist
    metric_map = 1 / (s * norm_inv_map + t)

    return metric_map


def estimate(inv_depth_maps, cleardata):
    depth_maps = []
    for (im_id, _), inv_depth in tqdm(zip(cleardata, inv_depth_maps), total=len(cleardata),
                                      desc='Metric estimation :'):
        seq_name = cleardata.seq_name
        metric_depth = metric(inv_depth, min_dist=opt.seq_info[seq_name]['min_dist'],
                                         max_dist=opt.seq_info[seq_name]['max_dist'])
        depth_maps.append(metric_depth)
        if opt.plot_metric_depth:
            plot.inverse_and_metric_depths(inv_depth, metric_depth, cleardata.plots_root, im_id)

    return np.array(depth_maps)
