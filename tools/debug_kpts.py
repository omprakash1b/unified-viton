"""Diagnose garment keypoints (the input to the TPS warp).

    python tools/debug_kpts.py --split test --n 5

Prints the raw JSON of one keypoint file, then for a few samples the x/y range of the in-shop (kc),
predicted (kc_hat) and worn (kc_gt) keypoints, and how many points each TPS segment gets with its
x/y spread. A segment whose x-spread is ~0 makes the TPS smear the garment into horizontal streaks.
Writes results/debug_kpts/<name>.png: cloth with kc (numbered) | person with kc_hat (green) and kc_gt (red).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vitonhd import VitonHD  # noqa: E402
from models.keypoint_gcn import denormalise_xy  # noqa: E402
from models.warp import assign_segments  # noqa: E402
from pipeline import Geometry, ckpt, load_cfg  # noqa: E402

SEG_NAMES = ["centre", "left-up", "left-low", "right-up", "right-low"]


def to_img(t):
    return cv2.cvtColor(((t.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def rng(name, p):
    v = p[(p >= 0).all(1)]
    if len(v) == 0:
        return f"  {name:7s} no valid points"
    return (f"  {name:7s} valid {len(v):2d}/{len(p)}   x {v[:, 0].min():6.1f}..{v[:, 0].max():6.1f}   "
            f"y {v[:, 1].min():6.1f}..{v[:, 1].max():6.1f}   unique x {len(np.unique(v[:, 0].round()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="results/debug_kpts")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    h, w = cfg["data"]["height"], cfg["data"]["width"]
    ds = VitonHD(cfg, args.split, paired=True)
    os.makedirs(args.out, exist_ok=True)

    name = ds.pairs[0][1].replace(".jpg", ".json")
    for folder in (ds.kpt_dirs["cloth"], ds.kpt_dirs["image"]):
        if folder is None:
            continue
        path = os.path.join(ds.root, folder, name)
        print(f"\n--- raw {folder}/{name} (first 300 chars) ---")
        if os.path.exists(path):
            raw = open(path).read()
            print(raw[:300])
            obj = json.loads(raw)
            if isinstance(obj, dict):
                print("KGI format, lists: " + ", ".join(f"{k}={len(v)} points" for k, v in obj.items() if isinstance(v, list)))
            else:
                print(f"plain array of shape {np.array(obj, dtype=np.float32).shape}")
        else:
            print("file not found")

    geo = Geometry(cfg).eval()
    has_kp = os.path.exists(ckpt(cfg, "kp"))
    if has_kp:
        geo.kp.load_state_dict(torch.load(ckpt(cfg, "kp"), map_location="cpu"))

    for i in np.linspace(0, len(ds) - 1, args.n).astype(int):
        b = {k: (v[None] if torch.is_tensor(v) else [v]) for k, v in ds[i].items()}
        with torch.no_grad():
            kc_n, kc_hat_n = geo.run_kp(b)
        kc, gt = b["kc"][0].numpy(), b["kc_gt"][0].numpy()
        hat = torch.where(kc_hat_n < -1.5, -1.0, denormalise_xy(kc_hat_n, h, w))[0].numpy()
        print(f"\n=== {b['name'][0]}  (image {w}x{h}, pixels)")
        print(rng("kc", kc))
        print(rng("kc_hat", hat) + ("" if has_kp else "   (no kp.pt: kc_hat = untrained)"))
        print(rng("kc_gt", gt))
        valid = (kc_n > -1.5).all(-1) & (kc_hat_n > -1.5).all(-1)
        seg = assign_segments(kc_n, valid)[0].numpy()
        for s, sn in enumerate(SEG_NAMES):
            pts = kc[(seg == s) & valid[0].numpy()]
            xs = f"x-spread {np.ptp(pts[:, 0]):5.1f}px  y-spread {np.ptp(pts[:, 1]):5.1f}px" if len(pts) else ""
            print(f"    segment {sn:9s}: {len(pts):2d} points  {xs}")
        cloth, person = to_img(b["cloth"][0]), to_img(b["person"][0])
        for j, (x, y) in enumerate(kc):
            if x >= 0:
                cv2.circle(cloth, (int(x), int(y)), 3, (0, 200, 0), -1)
                cv2.putText(cloth, str(j), (int(x) + 3, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 120, 0), 1)
        for p, c in ((gt, (0, 0, 255)), (hat, (0, 200, 0))):
            for x, y in p:
                if x >= 0:
                    cv2.circle(person, (int(x), int(y)), 3, c, -1)
        cv2.imwrite(os.path.join(args.out, b["name"][0].replace(".jpg", ".png")), np.hstack([cloth, person]))
    print(f"\nimages in {args.out}")


if __name__ == "__main__":
    main()
