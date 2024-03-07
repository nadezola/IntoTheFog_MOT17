import argparse
from pathlib import Path
from DepthEstimation import monocular_depth
from FogRendering import volumetric_fog


def parth_args():
    # more options in opt.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/mot17/val/clear/images/MOT17-02',
                        help='Path to the clear weather images')
    parser.add_argument('--out', default='outputs/mot17/val/MOT17-02',
                        help='Output path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parth_args()
    clear_path = Path(args.input)
    output_path = Path(args.out)

    print(f"Processing: {clear_path.name}")

    image_list = sorted(list(clear_path.glob('*')))
    depth_maps = monocular_depth.run(image_list=image_list,
                                     output_path=output_path,
                                     model_path="DepthEstimation/weights/dpt_beit_large_512.pt",
                                     model_type="dpt_beit_large_512")
    volumetric_fog.fog_rendering(image_list, depth_maps, output_path)




