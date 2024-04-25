import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np


def parth_args():
    # more options in opt.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=[
                                            #'data/mot17/val/clear/images/MOT17-02',
                                            'outputs/EXPS-FOG/mot17/val_final_temporal/MOT17-02/fog_homo/2',
                                            'outputs/EXPS-FOG/mot17/val_final_temporal/MOT17-02/fog_hetero_0.8/3'
                                            ],
                        help='List of img roots')

    parser.add_argument('--outvideo', default='outputs/EXPS-FOG/mot17/vis_paper/MOT17-02-fog2-collage.mp4',
                        help='Output video path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parth_args()
    im_roots = [Path(p) for p in args.input]
    out_video = args.outvideo
    im_percent = 0.25
    fps = 20
    im_start = 0
    im_numbers = 600

    im_lists = [sorted(list(im_root.glob('*'))) for im_root in im_roots]

    imsize = cv2.imread(str(im_lists[0][0])).shape[:2]
    newsize = tuple([int(im_percent * s) for s in imsize])[::-1]
    collage_size = (newsize[0] * len(im_lists), newsize[1])

    video_writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, collage_size)

    for i in tqdm(range(im_start, im_start + im_numbers)):
        im_paths = [im_list[i] for im_list in im_lists]
        ims = [cv2.resize(cv2.imread(str(im_path)), newsize) for im_path in im_paths]

        im_collage = np.concatenate(ims, axis=1)
        # cv2.imshow('Collage', im_collage)
        # cv2.waitKey(0)
        video_writer.write(im_collage)
    video_writer.release()
