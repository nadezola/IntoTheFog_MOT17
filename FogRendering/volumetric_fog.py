import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import logging

from perlin_numpy import generate_fractal_noise_2d
from FogRendering import atmospheric_light
from DepthEstimation.utils import normalize
import opt

logger = logging.getLogger('FogRendering')

def mkdir(path):
    if not path.exists():
        path.mkdir(parents=True)


def perlin_noise_map(img_shape, map_shape=(1024, 1024), res=(4, 4), octave=6, cloud_brightness=1):
    assert 0.3 <= cloud_brightness <= 1, "Cloud brightness should be in range [0.3, 1]"

    np.random.seed(0)
    noise = generate_fractal_noise_2d(map_shape, res, octave)

    h, w = img_shape
    base = max(h, w)
    noise = cv2.resize(noise, (base, base), interpolation=cv2.INTER_LINEAR)
    noise = noise[:h, :w]
    noise_normalize = normalize(noise, scope=(1-cloud_brightness, 1))

    return np.expand_dims(noise_normalize, axis=2)


def optical_model_homo(image, depth, intensity, L_inf):
    beta = opt.beta[intensity - 1]
    T = np.exp(-beta * depth)
    fog_homo_img = T * image + L_inf * (1 - T)

    return fog_homo_img


def optical_model_hetero(image, depth, intensity, L_inf, turbulence):
    beta = opt.beta[intensity - 1]
    T = np.exp(-(beta + 0.5) * depth * turbulence)
    fog_hetero_img = T * image + L_inf * (1 - T)

    return fog_hetero_img


def rendering(dataloader, fog_homo_path, fog_hetero_path):
    _, probe_img, probe_depth = dataloader[0]
    atm_light = atmospheric_light.horizon_intensity(probe_img, np.squeeze(probe_depth))
    logger.info(f'Atmospheric light={atm_light:.2f}')

    turbulence_map = perlin_noise_map(probe_img.shape[:2], (1024, 1024), (4, 4), 8,
                                      cloud_brightness=opt.cloud_brightness)
    for intensity in opt.intensity:
        if fog_homo_path is not None:
            save_path_homo = fog_homo_path / f'{intensity}'
            save_path_homo.mkdir(parents=True, exist_ok=True)

        if fog_hetero_path is not None:
            save_path_hetero = fog_hetero_path / f'{intensity}'
            save_path_hetero.mkdir(parents=True, exist_ok=True)

        for img_stem, img, depthmap in tqdm(dataloader, total=len(dataloader), desc=f'Fog {intensity} rendering :'):

            if fog_homo_path is not None:
                fog_homo_img = optical_model_homo(img, depthmap, intensity, atm_light)
                fog_homo_BGR = cv2.cvtColor(fog_homo_img.astype(np.float32), cv2.COLOR_RGB2BGR) * 255.0
                cv2.imwrite(str(save_path_homo / f'{img_stem}.jpg'), fog_homo_BGR)

            if fog_hetero_path is not None:
                fog_hetero_img = optical_model_hetero(img, depthmap, intensity, atm_light, turbulence_map)
                fog_hetero_BGR = cv2.cvtColor(fog_hetero_img.astype(np.float32), cv2.COLOR_RGB2BGR) * 255.0
                cv2.imwrite(str(save_path_hetero / f'{img_stem}.jpg'), fog_hetero_BGR)

    if opt.plot_turbulence_map:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(np.squeeze(turbulence_map, axis=2), cmap='gray')
        fig.savefig(fog_hetero_path / 'turbulence_map.png')
        plt.close()
