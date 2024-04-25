import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import logging

from perlin_numpy import generate_fractal_noise_2d
from FogRendering import atmospheric_light
from DepthEstimation.utils import normalize
from DepthEstimation.metric_depth import metric
import opt

logger = logging.getLogger('FogRendering')

def mkdir(path):
    if not path.exists():
        path.mkdir(parents=True)


def perlin_noise_map(shape, cloud_brightness=1):
    assert 0.3 <= cloud_brightness <= 1, "Cloud brightness should be in range [0.3, 1]"
    np.random.seed(100)
    noise = generate_fractal_noise_2d((640, 640), (4, 4), 5)
    noise = cv2.resize(noise, shape[::-1])
    noise_normalize = normalize(noise, scope=(1-cloud_brightness, 1))

    return np.expand_dims(noise_normalize, axis=2)


def optical_model(image, depth, intensity, L_inf):
    beta = opt.beta[intensity - 1]

    # Homogeneous Fog
    T = np.exp(-beta * depth)
    fog_homo_img = T * image + L_inf * (1 - T)

    # Heterogeneous Fog
    turbulence = perlin_noise_map(image.shape[:2],  cloud_brightness=opt.cloud_brightness)
    T = np.exp(-beta * depth * turbulence)
    fog_hetero_img = T * image + L_inf * (1 - T)

    return turbulence, fog_homo_img, fog_hetero_img


def rendering(cleardata, depth_maps):

    if opt.seq_info[cleardata.seq_name]['max_dist'] == 1e6:
        atm_light = atmospheric_light.horizon_intensity(image=cleardata[0], depth=depth_maps[0])
    else:
        atm_light = atmospheric_light.dark_channel(image=cleardata[0])
    logger.info(f'Atmospheric light={atm_light:.2f}')

    turbulence_map = None
    for intensity in opt.intensity:
        save_path_homo = cleardata.fog_homo_root / f'{intensity}'
        save_path_hetero = cleardata.fog_hetero_root / f'{intensity}'
        mkdir(save_path_homo)
        mkdir(save_path_hetero)

        for (im_id, im), depth in tqdm(zip(cleardata, depth_maps), total=len(cleardata),
                                       desc=f'Fog {intensity} rendering :'):
            depth = np.expand_dims(depth, axis=2)
            turbulence_map, fog_homo_im, fog_hetero_im = optical_model(im, depth, intensity, atm_light)
            fog_homo_BGR = cv2.cvtColor(fog_homo_im.astype(np.float32), cv2.COLOR_RGB2BGR) * 255.0
            fog_hetero_BGR = cv2.cvtColor(fog_hetero_im.astype(np.float32), cv2.COLOR_RGB2BGR) * 255.0
            cv2.imwrite(str(save_path_homo / f'{im_id}.jpg'), fog_homo_BGR)
            cv2.imwrite(str(save_path_hetero / f'{im_id}.jpg'), fog_hetero_BGR)

    out_root = cleardata.fog_homo_root.parent
    if opt.plot_turbulence_map:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(np.squeeze(turbulence_map, axis=2), cmap='gray')
        fig.savefig(out_root / 'turbulence_map.png')
        plt.close()

    logger.info(f'Fog augmentation results are saved in {str(out_root)}')
