import numpy as np
from tqdm import tqdm


def seq_depth(depth_list):
    x, y = 10, 10
    ref = np.mean(depth_list[:, y, x])
    consist_depths = np.empty_like(depth_list)
    for idx in tqdm(range(depth_list.shape[0]), desc="Temporal Consistency processing"):
        depth_map = depth_list[idx]
        scale = ref / depth_map[y, x]
        consist_depths[idx] = depth_map * scale

    return consist_depths
