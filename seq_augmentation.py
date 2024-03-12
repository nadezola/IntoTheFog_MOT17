import argparse
from pathlib import Path
from DepthEstimation import monocular_depth, metric_estimation
from FogRendering import volumetric_fog
import opt


def parth_args():
    # more options in opt.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/mot17/debug/clear/images',
                        help='Path to the clear weather images')
    parser.add_argument('--out', default='outputs/mot17/debug',
                        help='Output path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parth_args()
    clear_path = Path(args.input)
    output_path = Path(args.out)

    clear_folders = sorted(list(clear_path.glob('*')))
    for clear_folder in clear_folders:
        print(f"Processing: {clear_folder.name}")

        image_list = sorted(list(clear_folder.glob('*')))

        if opt.estimate_depth:
           # Estimate depth from  monocular images
           inv_depth_maps = monocular_depth.run(image_list=image_list,
                                                output_path=output_path / clear_folder.name,
                                                model_path="DepthEstimation/weights/dpt_beit_large_512.pt",
                                                model_type="dpt_beit_large_512")
        else:
           # Load depth from files
           inv_depth_maps = monocular_depth.load(image_list=image_list,
                                                 depth_path=output_path / clear_folder.name / 'inv_relative_depth')

        depth_maps = metric_estimation.run(inv_depth_maps,
                                           image_list=image_list,
                                           output_path=output_path / clear_folder.name,)

        volumetric_fog.fog_rendering(image_list, depth_maps, output_path / clear_folder.name)




