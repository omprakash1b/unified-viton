"""Check Stage 2 (segmentation + occlusion): numbers + overlay images.

    python tools/check_seg.py --split test --n 300 --save 20

Uses checkpoints/kp.pt (Stage 1 predictions feed Stage 2, as in training) and checkpoints/seg.pt.
Numbers are computed inside the agnostic mask M_a (outside it the parsing is given as input):
  per-class IoU, mIoU, garment IoU, arm IoU, and occlusion precision/recall/F1/IoU.
Hard-pose images (occluded area / garment area >= --thr) are also scored separately.
Images (results/check_seg/): person | agnostic | GT parsing | predicted parsing | occlusion (red GT, green pred, yellow both)
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vitonhd import SEG_UPPER, VitonHD  # noqa: E402
from pipeline import Geometry, ckpt, load_cfg, to_device  # noqa: E402

NAMES = ["bg", "hair", "face", "upper", "l-arm", "r-arm", "lower", "neck"]
COLORS = np.array([[0, 0, 0], [0, 85, 170], [170, 200, 255], [0, 0, 220], [0, 200, 0],
                   [200, 200, 0], [170, 0, 170], [0, 170, 255]], np.uint8)  # BGR


def to_img(t):
    return cv2.cvtColor(((t.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


class Scores:
    def __init__(self, n):
        self.inter, self.union = np.zeros(n), np.zeros(n)
        self.tp = self.fp = self.fn = 0
        self.imgs = 0

    def add(self, pred, gt, occ_p, occ_g, roi):
        for c in range(len(self.inter)):
            p, g = (pred == c) & roi, (gt == c) & roi
            self.inter[c] += (p & g).sum()
            self.union[c] += (p | g).sum()
        self.tp += (occ_p & occ_g).sum()
        self.fp += (occ_p & ~occ_g).sum()
        self.fn += (~occ_p & occ_g).sum()
        self.imgs += 1

    def report(self, title):
        if self.imgs == 0:
            return
        iou = self.inter / np.maximum(self.union, 1)
        present = self.union > 0
        prec = self.tp / max(self.tp + self.fp, 1)
        rec = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"\n[{title}]  images={self.imgs}")
        print("  IoU in M_a: " + "  ".join(f"{n} {v * 100:.1f}" for n, v, ok in zip(NAMES, iou, present) if ok))
        print(f"  mIoU {iou[present].mean() * 100:.1f}   garment {iou[SEG_UPPER] * 100:.1f}   "
              f"arms {(iou[4] + iou[5]) / 2 * 100:.1f}")
        print(f"  occlusion  P {prec * 100:.1f}  R {rec * 100:.1f}  F1 {f1 * 100:.1f}  "
              f"IoU {self.tp / max(self.tp + self.fp + self.fn, 1) * 100:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300, help="samples to score (0 = all)")
    ap.add_argument("--save", type=int, default=20, help="overlay images to write")
    ap.add_argument("--thr", type=float, default=0.08, help="hard-pose threshold: occluded / garment area")
    ap.add_argument("--out", default="results/check_seg")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    n_cls = cfg["model"]["num_seg_classes"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geo = Geometry(cfg).to(dev).eval()
    geo.load(cfg, "seg")  # kp.pt
    geo.seg.load_state_dict(torch.load(ckpt(cfg, "seg"), map_location="cpu"))

    ds = VitonHD(cfg, args.split, paired=True)
    if args.n:
        ds = Subset(ds, np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int).tolist())
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    os.makedirs(args.out, exist_ok=True)

    all_s, hard_s, saved = Scores(n_cls), Scores(n_cls), 0
    with torch.no_grad():
        for b in dl:
            b = to_device(b, dev)
            _, kc_hat = geo.run_kp(b)
            seg_logits, occ_logit = geo.run_seg(b, kc_hat)
            pred = seg_logits.argmax(1).cpu().numpy()
            occ_p = (torch.sigmoid(occ_logit[:, 0]) > 0.5).cpu().numpy()
            gt = b["seg"].cpu().numpy()
            occ_g = b["occ"][:, 0].cpu().numpy() > 0.5
            roi = b["mask_a"][:, 0].cpu().numpy() > 0.5

            for i in range(len(pred)):
                all_s.add(pred[i], gt[i], occ_p[i], occ_g[i], roi[i])
                garment = (gt[i] == SEG_UPPER).sum() + occ_g[i].sum()
                if garment > 0 and occ_g[i].sum() / garment >= args.thr:
                    hard_s.add(pred[i], gt[i], occ_p[i], occ_g[i], roi[i])

                if saved < args.save:
                    person = to_img(b["person"][i])
                    occ_vis = (person * 0.4).astype(np.uint8)
                    occ_vis[occ_g[i] & ~occ_p[i]] = (0, 0, 255)
                    occ_vis[occ_p[i] & ~occ_g[i]] = (0, 220, 0)
                    occ_vis[occ_p[i] & occ_g[i]] = (0, 255, 255)
                    panels = [person, to_img(b["agnostic"][i]), COLORS[gt[i]], COLORS[pred[i]], occ_vis]
                    for p in panels[2:4]:  # outline M_a so you can see which region was predicted
                        cnts, _ = cv2.findContours(roi[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(p, cnts, -1, (255, 255, 255), 1)
                    cv2.imwrite(os.path.join(args.out, b["name"][i].replace(".jpg", "_seg.png")), np.hstack(panels))
                    saved += 1

    all_s.report(f"{args.split}: all")
    hard_s.report(f"{args.split}: hard pose (occlusion >= {args.thr:.2f})")
    print(f"\noverlays: {saved} images in {args.out}")
    print("legend: " + ", ".join(f"{n}={tuple(int(x) for x in c[::-1])}" for n, c in zip(NAMES, COLORS)) + " (RGB)")


if __name__ == "__main__":
    main()