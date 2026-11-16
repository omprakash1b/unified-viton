"""Check Stage 1 (keypoint GCN): error numbers + overlay images.

    python tools/check_kp.py --split test --n 300 --save 20

Numbers (at the training resolution, pixels):
  baseline  = error if the garment keypoints are not moved at all (input kc vs ground truth)
  predicted = error of the GCN output kc_hat vs ground truth
  PCK@t     = share of keypoints within t pixels of ground truth
Images (results/check_kp/): cloth with its keypoints | person with GT (red), prediction (green), pose (blue).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vitonhd import VitonHD  # noqa: E402
from models.keypoint_gcn import denormalise_xy  # noqa: E402
from pipeline import Geometry, ckpt, load_cfg, to_device  # noqa: E402

# nose, neck, r-sho, r-elb, r-wri, l-sho, l-elb, l-wri, r-hip, l-hip
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (1, 9)]


def to_img(t):
    return cv2.cvtColor(((t.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def dots(img, xy, color, r=3, label=False):
    for i, (x, y) in enumerate(xy):
        if x < 0 or y < 0:
            continue
        cv2.circle(img, (int(x), int(y)), r, color, -1)
        if label:
            cv2.putText(img, str(i), (int(x) + 3, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300, help="samples to score (0 = all)")
    ap.add_argument("--save", type=int, default=20, help="overlay images to write")
    ap.add_argument("--out", default="results/check_kp")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    h, w = cfg["data"]["height"], cfg["data"]["width"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geo = Geometry(cfg).to(dev).eval()
    geo.kp.load_state_dict(torch.load(ckpt(cfg, "kp"), map_location="cpu"))

    ds = VitonHD(cfg, args.split, paired=True)
    if args.n:
        ds = Subset(ds, np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int).tolist())
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    os.makedirs(args.out, exist_ok=True)

    err_base, err_pred, saved, no_gt = [], [], 0, 0
    with torch.no_grad():
        for b in dl:
            b = to_device(b, dev)
            kc, kc_hat = geo.run_kp(b)
            gt = b["kc_gt"]
            pred_px = torch.where(kc_hat < -1.5, -1.0, denormalise_xy(kc_hat, h, w))
            valid = (gt >= 0).all(-1) & (b["kc"] >= 0).all(-1)  # [B,N]
            no_gt += int((~(gt >= 0).all(-1)).all(-1).sum())
            err_base.append((b["kc"] - gt).norm(dim=-1)[valid].cpu())
            err_pred.append((pred_px - gt).norm(dim=-1)[valid].cpu())

            for i in range(len(b["name"])):
                if saved >= args.save:
                    break
                cloth, person = to_img(b["cloth"][i]), to_img(b["person"][i])
                dots(cloth, b["kc"][i].cpu().numpy(), (0, 200, 0), label=True)
                pose = b["pose"][i].cpu().numpy()
                for a, c in BONES:
                    if pose[a, 2] > 0 and pose[c, 2] > 0:
                        cv2.line(person, tuple(pose[a, :2].astype(int)), tuple(pose[c, :2].astype(int)), (255, 120, 0), 2)
                dots(person, gt[i].cpu().numpy(), (0, 0, 255), 3)
                dots(person, pred_px[i].cpu().numpy(), (0, 220, 0), 2, label=True)
                for p, g in zip(pred_px[i].cpu().numpy(), gt[i].cpu().numpy()):  # yellow = error
                    if (p >= 0).all() and (g >= 0).all():
                        cv2.line(person, tuple(p.astype(int)), tuple(g.astype(int)), (0, 255, 255), 1)
                cv2.imwrite(os.path.join(args.out, b["name"][i].replace(".jpg", "_kp.png")), np.hstack([cloth, person]))
                saved += 1

    eb, ep = torch.cat(err_base), torch.cat(err_pred)
    if ep.numel() == 0:
        print(f"No ground-truth garment keypoints found in '{args.split}' (image-cloth-kpts/). "
              f"Images were still saved to {args.out}; use --split train to get numbers.")
        return
    print(f"split={args.split}  points scored={ep.numel()}  images without GT={no_gt}  res={h}x{w}")
    print(f"mean error  baseline {eb.mean():6.2f}px   predicted {ep.mean():6.2f}px   median {ep.median():.2f}px")
    for t in (5, 10, 20):
        print(f"PCK@{t:<2}px    baseline {(eb < t).float().mean() * 100:5.1f}%   predicted {(ep < t).float().mean() * 100:5.1f}%")
    print(f"overlays: {saved} images in {args.out}")


if __name__ == "__main__":
    main()