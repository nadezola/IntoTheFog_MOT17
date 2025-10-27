from pathlib import Path
from typing import Union
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
from tqdm import tqdm


def run(seq: Path, out_root:  Union[Path, str]) -> None:
    out_root = Path(out_root)
    depth_dir = out_root / "depthmaps"
    depth_dir.mkdir(parents=True, exist_ok=True)

    depth_estimator = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf"
    )

    img_root = seq / "img1"
    imgs = sorted(list(img_root.glob("*.jpg")))
    for img_p in tqdm(imgs, total=len(imgs)):
        img = Image.open(img_p)
        depthpil_255 = depth_estimator(img)['depth']
        depthmap_255 = np.array(depthpil_255, dtype=np.uint8)
        cv2.imwrite(str(depth_dir / f"{img_p.stem}.png"), depthmap_255)
