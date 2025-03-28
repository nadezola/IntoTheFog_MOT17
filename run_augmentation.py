import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

from dataprocess import ClearDepthDataset
from DepthEstimation import monocular_depth, metric_depth, temporal
from FogRendering import volumetric_fog
import opt

logger = logging.getLogger('FogRendering')
logging.basicConfig(level=logging.INFO,
                    format='[Info] :: %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('log.txt')])


def parth_args():
    # more options in opt.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/media/nadesha/hdd/SAFER/Frequentis/SAFER-v5/tracker-05-external-small',
                        help='Root of clear images')
    parser.add_argument('--out', default='/media/nadesha/hdd/SAFER-exps/Fog/images',
                        help='Output root')
    parser.add_argument('--loaddepth', default='/media/nadesha/hdd/SAFER-exps/Adaptation/depth_estimation/exp-1',
                        help='Load depth images instead of depth estimation')
    # parser.add_argument('--input', default='/media/nadesha/hdd/SAFER-exps/AIT/weather_images/clear',
    #                     help='Root of clear images')
    # parser.add_argument('--out', default='/media/nadesha/hdd/SAFER-exps/AIT/weather_images/foggy',
    #                     help='Output root')
    # parser.add_argument('--loaddepth', default='/media/nadesha/hdd/SAFER-exps/AIT/weather_images/depth',
    #                     help='Load depth images instead of depth estimation')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parth_args()
    clear_root = Path(args.input)
    depth_root = Path(args.loaddepth) if args.loaddepth != '' else None
    out_root = Path(args.out)

    # clear_seqs = sorted(list(clear_root.glob('*')))
    # clear_seqs = [
    #     # clear_root / 'MOT17-02',
    #     # clear_root / 'MOT17-04',
    #     # clear_root / 'MOT17-05',
    #     # clear_root / 'MOT17-09',
    #     # clear_root / 'MOT17-10',
    #     # clear_root / 'MOT17-11',
    #     # clear_root / 'MOT17-13',
    # ]

    seq_names = {
        'test-similar-ap': [
            # 'da99e38330a77c46b0d7328a9efde2be28ac63dcc66821cfdfa9148211d8b9a9',
            'c26030b369403d08d30b2f03b6ec145edf6109c949a598c1e5088e0271488ed0',
            'f6fcf9b6aa7feda09c97d6df1598d390563cccb657a92a8d4a2e1ae3285500fe',
            '64fb9296421d0d217dba5b3a2d0f8d90600021c4b9f581177cef355bc9a824a4',
            '92827ea9c58065580b5cb167b2f23e375b06495db615b24dda27422469159177',
            'ed61b4f5831c47f1ca57a641d26391dfa36ea29564db096b9c1ffa25372b02a2',
            '1c3c4effd6747e14f8d46706a80b8dc78d170360bd377bb0f263cb943e714fa1',
            '1e3c094b88dffcd99fdeb905d643e2b6ab6cda74fabe2e10156e584ad61f743c',
            '5b8013919b2dfc4f9ac2b1865cccaed4a3183068765e287db8dfefbaac578db9',
            '57eb6c21ff875b22b30acb25d3c21eec4202790a1fc521d9239970573e706ccf',
            '7881b57bc79e890ff70bbabf7c48f59b6bf7bcf0865fa7ad0d2224406a299a2c',
            'b6e61ab2b9ac40614fbc2a0416192c41acde90ea0c84ce8b8e86189b8e161335',
            'b9121e814a0a044435cb72453fbe24dfe38d1cd3aab55069f00b516d5e2a6969',
            'f3c12382d8cb2ea3c3523ec229fe76ae536b66f45268542a07a922590efea91e',
        ],

        # 'test-diff-ap': [
        #     '0b1b11a9e05ccc6f03fd74b2b84470e52625f038f46a9d18f382673fe72fd0b8',
        #     '1cf8704dbc416ad4f8f4864fdd2f08cde31f52d782edc3d148e0dfc21f6f7c0d',
        #     '1e2f98ff9c6a8b354cdd1f77029a24e441e28cf3afc4e1cf54c76f9ae0531729',
        #     '5b2b375b15712451b6381060cd53ade6fb3bbbe61db91b6adb4240bfd258809a',
        #     '15bbf60ee58322a59a92fb0c2579d4edd09beb354434e20f568017c0797e8f59',
        #     '361e5263b001ff5a4fa556b3e4c7b8b2a66cbe28e2a434253454cd35fa2f0bbe',
        #     '595f7d9abb965a0794a4a93429a235be349b8f073afece81538c8bd98882b67c',
        #     'aeb1ea21927eb8f561c002319cb9b113d63adc6f3bd6715e82fe57e15823d6e8',
        #     'c23d99b231b6c3bd2b2bf119f8f2487490256e9fcbbca6c6741070931cd3286c',
        #     'c331013b0d71316e2ec74e207166849703059a919776c306ffa2913ef78a9444',
        #     'd263ef1f6d9d78e65ef1ad9959f8a9053c41f2d6153c7ceb26f1b019e29eeeac',
        #     'dc98e20dee2f59f28abea3ade17ba11989e398313a6e1588b100f2b2d3d993f5',
        #     'e6bccbddd92318fb1a100c2da5ba8c921618cbb271c254ab42135f0f48d52c26',
        #     'e07d524e0a66ccc65735c6ed4b0453f4d2bd2a6794a9403bc148b5eab3231860'
        # ]
    }

    clear_seqs = []
    for key, value in seq_names.items():
        for seq in value:
            clear_seqs.append(clear_root / key / seq)

    for seq in clear_seqs:
        logger.info(f'{(datetime.now()).strftime("%d-%m-%Y %H:%M:%S")}')
        logger.info(f"Processing {seq.parent.name}/{seq.name} sequence")

        seq_out_root = out_root / seq.parent.name / seq.name
        fog_homo_path = seq_out_root / 'fog_homo'
        fog_hetero_path = seq_out_root / f'fog_hetero_{opt.cloud_brightness}'

        # if args.loaddepth == '':
        #     # TODO
        #     estimated_depth_path = seq_out_root / 'depthmaps'
        #     depth_color_path = seq_out_root / 'depthmaps_color'
        #     plots_path = seq_out_root / 'depth_metric'
        #     depth_root = monocular_depth.estimate(cleardata=seq,
        #                                           model_path="DepthEstimation/weights/dpt_beit_large_512.pt",
        #                                           model_type="dpt_beit_large_512")

        dataloader = ClearDepthDataset(args, seq, depth_root)

        #aligned_depth_maps = temporal.align_depth_maps(pred_inv_depth_maps)
        # aligned_depth_maps = pred_inv_depth_maps
        # depth_maps = metric_depth.estimate(inv_depth_maps=aligned_depth_maps, cleardata=clr_data)
        # volumetric_fog.rendering(clr_data, depth_maps)
        volumetric_fog.rendering(dataloader, fog_homo_path, fog_hetero_path)