import argparse
from pathlib import Path
from DepthEstimation import monocular_depth, temporal_consistency
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

    image_list = sorted(list(clear_path.glob('*')))
    depth_maps = monocular_depth.run(image_list=image_list,
                                     model_path="DepthEstimation/weights/dpt_large_384.pt",
                                     model_type="dpt_large_384")
    cosistent_depth_maps = temporal_consistency.seq_depth(depth_maps)
    monocular_depth.save_depth(depth_maps, image_list, output_path)
    volumetric_fog.fog_rendering(image_list, depth_maps, output_path)




