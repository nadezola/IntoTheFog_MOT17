import numpy as np
import cv2
from tqdm import tqdm
from FogRendering import atmospheric_light
import opt


def fog_optical_model(image, depth, thickness, atm_light):
    beta = opt.beta[thickness - 1]
    T = np.exp(-beta*depth)
    L_inf = np.mean(atm_light)
    fog_img = T * image + L_inf * (1-T)

    return fog_img


def fog_rendering(image_list, depth_maps, output_path):
    atm_light = atmospheric_light.calculate_seq_intensity(image_list)

    fog_path = output_path / 'fog'
    for thickness in opt.THICKNESS:
        save_path = fog_path / str(thickness)
        if not save_path.exists():
            save_path.mkdir(parents=True)

        for idx in tqdm(range(len(image_list)),
                        desc=f'Fog Rensering of thickness {thickness}'):
            img = cv2.cvtColor(cv2.imread(str(image_list[idx])), cv2.COLOR_BGR2RGB) / 255.0
            depth = np.expand_dims(depth_maps[idx], axis=2) * opt.depth_scale
            foggy_img = fog_optical_model(img, depth, thickness, atm_light)
            cv2.imwrite(str(save_path / image_list[idx].name), foggy_img * 255)

    print(f'Fog augmentation saved in: {str(fog_path)}')
