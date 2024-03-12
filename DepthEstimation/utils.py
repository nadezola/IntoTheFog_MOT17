import numpy as np
import cv2


def normalize(values, scope=(0, 1)):
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite values present")

    values_min = values.min()
    values_max = values.max()

    if values_max - values_min > np.finfo("float").eps:
        scale = (scope[1] - scope[0]) / (values_max - values_min)
        values_normalized = scale * (values - values_min) + scope[0]
    else:
        values_normalized = np.zeros(values.shape, dtype=values.dtype)

    return values_normalized


def save_norm_depth(norm_depth, output_path, file_name):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # 16-bit encoding (https://developers.google.com/depthmap-metadata/encoding)
    max_value = 2**16-1
    norm_depth = max_value * norm_depth
    inv_depth_file = str(output_path / file_name) + '.png'
    cv2.imwrite(inv_depth_file, norm_depth.astype(np.uint16))


def load_norm_depth(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    return depth

