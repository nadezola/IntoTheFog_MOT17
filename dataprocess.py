import cv2
from pathlib import Path
import opt


class ClearDataset(object):
    def __init__(self, args, clr_folder):
        self.clr_imgs = sorted(list(clr_folder.glob('*')))
        if len(self.clr_imgs) == 0:
            raise FileNotFoundError("No images found")

        self.seq_name = clr_folder.name
        im = cv2.imread(str(self.clr_imgs[0]))
        self.img_size = im.shape[:-1]
        self.idx = 0

        # Output paths
        self.out_root = Path(args.out)
        self.depth_root = self.out_root / self.seq_name / 'depth_pred'
        self.depth_cl_root = self.out_root / self.seq_name / 'depth_color'
        self.plots_root = self.out_root / self.seq_name / 'depth_metric'
        self.fog_homo_root = self.out_root / self.seq_name / 'fog_homo'
        self.fog_hetero_root = self.out_root / self.seq_name / f'fog_hetero_{opt.cloud_brightness}'


    def __len__(self):
        return len(self.clr_imgs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.clr_imgs):
            self.idx = 0
            raise StopIteration

        im_id = self.clr_imgs[self.idx].stem
        im = cv2.cvtColor(cv2.imread(str(self.clr_imgs[self.idx])), cv2.COLOR_BGR2RGB) / 255.0
        self.idx += 1

        return im_id, im

    def __getitem__(self, i):
        im = cv2.cvtColor(cv2.imread(str(self.clr_imgs[i])), cv2.COLOR_BGR2RGB) / 255.0
        return im


