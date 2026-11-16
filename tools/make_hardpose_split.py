"""Build the hard-pose test split: images where arms cover a large part of the garment.

    python tools/make_hardpose_split.py --root ./data/VITON-HD --thr 0.08
Writes root/test_pairs_hardpose.txt (same format as test_pairs.txt). Rename/point the loader at it
(e.g. copy to a 'hardpose' split folder or swap the pairs file) to evaluate on it.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vitonhd import SEG_UPPER, group_parse, occlusion_from_parse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/dataset")
    ap.add_argument("--thr", type=float, default=0.08, help="occluded area / garment area")
    args = ap.parse_args()
    keep = []
    with open(os.path.join(args.root, "test_pairs.txt")) as f:
        pairs = [line.split() for line in f if line.strip()]
    for p, c in pairs:
        seg = np.array(Image.open(os.path.join(args.root, "test", "image-parse-v3", p.replace(".jpg", ".png"))))
        seg = group_parse(cv2.resize(seg, (384, 512), interpolation=cv2.INTER_NEAREST))
        occ = occlusion_from_parse(seg).sum()
        garment = (seg == SEG_UPPER).sum() + occ
        if garment > 0 and occ / garment >= args.thr:
            keep.append(f"{p} {c}\n")
    out = os.path.join(args.root, "test_pairs_hardpose.txt")
    with open(out, "w") as f:
        f.writelines(keep)
    print(f"{len(keep)} / {len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
