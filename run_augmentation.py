import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

from dataprocess import ClearDataset
from DepthEstimation import monocular_depth, metric_depth
from FogRendering import volumetric_fog
import opt

logger = logging.getLogger('FogRendering')
logging.basicConfig(level=logging.INFO,
                    format='[Info] :: %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('log.txt')])

def parth_args():
    # more options in opt.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/mot17/train/clear/images',
                        help='Root of clear images')
    parser.add_argument('--out', default='outputs/EXPS-FOG/mot17/train',
                        help='Output root')
    parser.add_argument('--loaddepth', action='store_true',
                        help='Load depth images instead of depth estimation')

    args = parser.parse_args()
    return args


def check_path(folder):
    if not folder.is_dir():
        raise NotADirectoryError(folder)


if __name__ == '__main__':
    args = parth_args()
    clear_root = Path(args.input)
    check_path(clear_root)

    # clear_folders = sorted(list(clear_root.glob('*')))
    clear_folders = [
        clear_root / 'MOT17-02',
        clear_root / 'MOT17-04',
        clear_root / 'MOT17-05',
        clear_root / 'MOT17-09',
        clear_root / 'MOT17-10',
        clear_root / 'MOT17-11',
        clear_root / 'MOT17-13'
    ]
    for clr_folder in clear_folders:
        logger.info(f'{(datetime.now()).strftime("%d-%m-%Y %H:%M:%S")}')
        logger.info(f"Processing {clr_folder.name} sequence")

        clr_data = ClearDataset(args, clr_folder)

        if args.loaddepth:
            pred_inv_depth_maps = monocular_depth.load(cleardata=clr_data,
                                                       depth_path=Path(args.out) / clr_folder.name / 'depth_pred')
        else:
            pred_inv_depth_maps = monocular_depth.estimate(cleardata=clr_data,
                                                           model_path="DepthEstimation/weights/dpt_beit_large_512.pt",
                                                           model_type="dpt_beit_large_512")
        depth_maps = metric_depth.estimate(inv_depth_maps=pred_inv_depth_maps, cleardata=clr_data)
        volumetric_fog.rendering(clr_data, depth_maps)
