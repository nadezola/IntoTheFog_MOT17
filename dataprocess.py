from pathlib import Path
import numpy as np
import cv2
import opt


class ClearDepthDataset(object):
    def __init__(self, args, seq_path, depth_root):
        self.idx = -1

        img_folder = seq_path / "img1"
        self.clr_imgs = sorted(list(img_folder.glob('*')))
        if len(self.clr_imgs) == 0:
            raise FileNotFoundError("No images found")
        im = cv2.imread(str(self.clr_imgs[0]))
        self.img_size = im.shape[:-1]

        self.seq_type = seq_path.parent.name
        self.seq_name = seq_path.name

        self.depth_root = Path(depth_root) / self.seq_type / self.seq_name / "depthmaps"

    def __len__(self):
        return len(self.clr_imgs)

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= len(self.clr_imgs):
            self.idx = -1
            raise StopIteration

        img_stem, img_norm, depthmap = self.__getitem__(self.idx)

        return img_stem, img_norm, depthmap

    def __getitem__(self, i):
        img_stem = self.clr_imgs[i].stem

        img = cv2.imread(str(self.clr_imgs[i]))
        img_norm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        depth_file = self.depth_root / f'{img_stem}.png'
        if not depth_file.exists():
            raise FileNotFoundError(f'For image {img_stem} the depth map {depth_file} does not exist!')

        depthmap_255 = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
        depthmap = 1 - depthmap_255 / 255.0
        depthmap = np.expand_dims(depthmap, axis=2)

        return img_stem, img_norm, depthmap


