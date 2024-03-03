import argparse
from pathlib import Path
from DepthEstimation import monocular_depth
from FogRendering import volumetric_fog


def parth_args():
    # more options in opt.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/mot17/debug/clear/images/MOT17-02',
                        help='Path to the clear weather images')
    parser.add_argument('--out', default='outputs/mot17/debug/MOT17-02',
                        help='Output path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parth_args()
    clear_path = Path(args.input)
    output_path = Path(args.out)

    print(f"Processing: {clear_path.name}")

    # from perlin_numpy import generate_fractal_noise_2d
    # from matplotlib import pyplot as plt
    # from DepthEstimation.monocular_depth import normalize
    # import numpy as np
    # import cv2
    # np.random.seed(111)
    # noise = generate_fractal_noise_2d((1024, 1280), (4, 4), 5)
    # noise = cv2.resize(noise, (1920, 1080))
    # noise = normalize(noise)
    # noise = (noise * 255).astype(np.uint8)
    # plt.figure()
    # plt.imshow(noise, cmap='gray')
    # plt.show()
    # plt.close()
    # image_list = sorted(list(clear_path.glob('*')))
    # img = cv2.imread(str(image_list[0]))
    # #img = cv2.cvtColor(cv2.imread(str(image_list[0])), cv2.COLOR_BGR2GRAY)
    # noise = cv2.merge([noise, noise, noise])
    # fog_img = cv2.addWeighted(img, 0.5, noise, 0.5, 0)
    # cv2.imshow('Noise', noise)
    # cv2.imshow('Foggy image', fog_img)
    # cv2.waitKey(0)

    image_list = sorted(list(clear_path.glob('*')))
    depth_maps = monocular_depth.run(image_list=image_list,
                                     output_path=output_path,
                                     model_path="DepthEstimation/weights/dpt_beit_large_512.pt",
                                     model_type="dpt_beit_large_512")
    volumetric_fog.fog_rendering(image_list, depth_maps, output_path)




