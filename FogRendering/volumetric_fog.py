import numpy as np
import cv2
from tqdm import tqdm
from FogRendering import atmospheric_light
from DepthEstimation.monocular_depth import normalize, metric
from perlin_numpy import generate_fractal_noise_2d
from matplotlib import pyplot as plt
import opt


def perlin_noise(shape):
    np.random.seed(111)
    noise = generate_fractal_noise_2d((1024, 1280), (4, 4), 5)
    noise = cv2.resize(noise, shape[::-1])
    noise_normalize = normalize(noise)
    noise_metric = metric(noise_normalize)

    # Plot Noise
    # plt.figure()
    # plt.imshow(noise_metric, cmap='gray')
    # plt.show()
    # plt.close()

    return np.expand_dims(noise_metric, 2)


def fog_optical_model(image, depth, thickness, atm_light):
    beta = opt.beta[thickness - 1]
    turbulence = perlin_noise(image.shape[:2])

    # Homogeneous fog
    # T = np.exp(-beta * depth)
    # or
    # T = np.exp(-np.power(beta * depth, 2))

    # Heterogeneous fog
    T = np.exp(-beta * depth * turbulence)
    L_inf = np.mean(atm_light)
    fog_img = T * image + L_inf * (1-T)

    return fog_img


def fog_rendering(image_list, depth_maps, output_path):
    atm_light = atmospheric_light.horizon_intensity(image_list, depth_maps, 0.9*opt.max_dist)
    #atm_light = atmospheric_light.image_intensity(image_list)
    #atm_light = 0.8

    fog_path = output_path / 'fog'
    for thickness in opt.THICKNESS:
        save_path = fog_path / str(thickness)
        if not save_path.exists():
            save_path.mkdir(parents=True)

        for idx in tqdm(range(len(image_list)),
                        desc=f'Fog Rensering of thickness {thickness}'):
            img = cv2.cvtColor(cv2.imread(str(image_list[idx])), cv2.COLOR_BGR2RGB) / 255.0
            depth = np.expand_dims(depth_maps[idx], axis=2)
            foggy_img = fog_optical_model(img, depth, thickness, atm_light)
            foggy_img_bgr = cv2.cvtColor(foggy_img.astype(np.float32), cv2.COLOR_RGB2BGR) * 255.0
            cv2.imwrite(str(save_path / image_list[idx].name), foggy_img_bgr)

    print(f'Fog augmentation saved in: {str(fog_path)}')
