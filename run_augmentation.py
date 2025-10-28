import argparse
from pathlib import Path
import logging
from datetime import datetime

from dataprocess import ClearDepthDataset
from DepthEstimation import depth_estimation
from FogRendering import volumetric_fog
import opt

logger = logging.getLogger('FogRendering')
logging.basicConfig(level=logging.INFO,
                    format='[Info] :: %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('log.txt')])


def parth_args():
    # more options in opt.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='Billund-Dataset/test',
                        help='Root of a data split')
    parser.add_argument('--out', default='Billund-Dataset/foggy',
                        help='Output root')
    parser.add_argument('--loaddepth', default='',
                        help='(Optional) Load depth images instead of depth estimation')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parth_args()
    clear_root = Path(args.input)
    depth_exists = True if args.loaddepth != '' else False
    out_root = Path(args.out)
    fog_types = opt.fog_types

    clear_seqs = sorted(list(clear_root.glob('*')))
    for seq in clear_seqs:
        logger.info(f'{(datetime.now()).strftime("%d-%m-%Y %H:%M:%S")}')
        logger.info(f"Processing {seq.parent.name}/{seq.name} sequence")

        seq_out_root = out_root / seq.parent.name / seq.name
        fog_homo_path = seq_out_root / 'fog_homo' if "homo" in fog_types else None
        fog_hetero_path = seq_out_root / f'fog_hetero_{opt.cloud_brightness}' if "hetero" in fog_types else None

        if not depth_exists:
            depth_estimation.run(seq, seq_out_root)
            depth_root = out_root
        else:
            depth_root = Path(args.loaddepth)

        dataloader = ClearDepthDataset(args, seq, depth_root)
        volumetric_fog.rendering(dataloader, fog_homo_path, fog_hetero_path)