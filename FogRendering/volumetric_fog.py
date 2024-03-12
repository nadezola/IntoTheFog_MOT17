import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

from perlin_numpy import generate_fractal_noise_2d

from FogRendering import atmospheric_light
from DepthEstimation.utils import normalize
from DepthEstimation.metric_estimation import metric

import opt


def perlin_noise_map(shape, cloud_brightness=1, plot=False):
    np.random.seed(100)
    noise = generate_fractal_noise_2d((640, 640), (4, 4), 5)
    noise = cv2.resize(noise, shape[::-1])
    noise_normalize = normalize(noise, scope=(1-cloud_brightness, 1))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(noise_normalize, cmap='gray')
        plt.show()
        plt.close()

    return noise_normalize


def fog_optical_model(image, depth, thickness, atm_light, heterogeneous, seq_name, cloud_brightness=1, plot=False):
    beta = opt.beta[thickness]

    if heterogeneous:
        turbulence_texture = perlin_noise_map(image.shape[:2], cloud_brightness, plot=plot)
        turbulence = metric(turbulence_texture,
                            min_dist=opt.seq_info[seq_name]['min_dist'],
                            max_dist=opt.seq_info[seq_name]['max_dist'])
        turbulence = np.expand_dims(turbulence, 2)
        T = np.exp(-beta * depth * turbulence)
    else:
        T = np.exp(-beta * depth)
        # T = np.exp(-np.power(beta * depth, 2))

    L_inf = np.mean(atm_light)
    fog_img = T * image + L_inf * (1-T)

    return fog_img


def fog_rendering(image_list, depth_maps, output_path):
    seq_name = image_list[0].parent.stem
    heterogeneous = opt.heterogeneous_fog

    # Atmospheric light
    if opt.seq_info[seq_name]['horizon']:
        atm_light = atmospheric_light.horizon_intensity(image_list, depth_maps, 0.9*opt.seq_info[seq_name]['max_dist'])
    else:
        atm_light = atmospheric_light.image_intensity(image_list)
        #atm_light = 0.8

    # Check Heterogeneous
    if heterogeneous:
        cloud_brightness = opt.cloud_brightness
        if 0.3 <= cloud_brightness <= 1:
            fog_path = output_path / f'fog_heterogeneous_{cloud_brightness}'
        else:
            heterogeneous = False
            print('WARNING: Cloud brightness is not in range [0.3, 1]. Homogeneous fog will be rendered.')

    if not heterogeneous:
        fog_path = output_path / 'fog_homogeneous'

    # Fog Rendering
    for thickness, visib in enumerate(opt.visibility):
        save_path = fog_path / f'{visib}'
        if not save_path.exists():
            save_path.mkdir(parents=True)

        for idx in tqdm(range(len(image_list)),
                        desc=f'Fog Rendering {visib}'):

            img = cv2.cvtColor(cv2.imread(str(image_list[idx])), cv2.COLOR_BGR2RGB) / 255.0
            depth = np.expand_dims(depth_maps[idx], axis=2)

            if thickness == 0 and idx == 0 and opt.plot_turbulence_map:
                foggy_img = fog_optical_model(img, depth, thickness, atm_light, heterogeneous, seq_name,
                                              cloud_brightness=cloud_brightness,
                                              plot=True)
            else:
                foggy_img = fog_optical_model(img, depth, thickness, atm_light, heterogeneous, seq_name,
                                              cloud_brightness=cloud_brightness)

            foggy_img_bgr = cv2.cvtColor(foggy_img.astype(np.float32), cv2.COLOR_RGB2BGR) * 255.0
            cv2.imwrite(str(save_path / image_list[idx].name), foggy_img_bgr)

    print(f'Fog augmentation saved in: {str(fog_path)}')
