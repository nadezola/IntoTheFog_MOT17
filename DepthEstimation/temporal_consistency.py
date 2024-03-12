import torch
import numpy as np


def align_depth_maps(map, ref_map):
    grid_step = 100
    map_grid = map[::grid_step, ::grid_step].flatten()
    ref_grid = ref_map[::grid_step, ::grid_step].flatten()

    A = np.vstack([map_grid, np.ones(len(map_grid))]).T
    s, t = np.linalg.lstsq(A, ref_grid, rcond=None)[0]

    aligned_map = s * map + t

    return aligned_map


def align(prediction, target):
    mask = target == target
    mask = torch.from_numpy(mask).unsqueeze(0)
    prediction = torch.from_numpy(prediction).unsqueeze(0)
    target = torch.from_numpy(target).unsqueeze(0)
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    s, t = x_0, x_1
    aligned_prediction = s * prediction + t
    aligned_prediction1 = s.view(-1, 1, 1) * prediction + t.view(-1, 1, 1)

    return x_0, x_1