"""Per-problem evaluation (doc section "Experimental setup").

    python -m eval.metrics --pred results/paired   --split test --paired     # SSIM, LPIPS, garment LPIPS
    python -m eval.metrics --pred results/unpaired --split test              # FID, KID, skin ΔE
Optional: --pose_json DIR with OpenPose re-detections of the outputs -> PCK@0.05.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data.vitonhd import BODY25_IDX, COCO18_IDX, SEG_FACE, SEG_LARM, SEG_RARM, SEG_UPPER, group_parse


def load(path, hw=None):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if hw:
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return img


def to_t(img):
    return torch.from_numpy(img).permute(2, 0, 1).float()[None] / 127.5 - 1


def skin_delta_e(img, seg):
    """ΔE (CIELAB) between mean colour of generated arms and visible face skin."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    arms, face = np.isin(seg, [SEG_LARM, SEG_RARM]), seg == SEG_FACE
    if arms.sum() < 50 or face.sum() < 50:
        return None
    return float(np.linalg.norm(lab[arms].mean(0) - lab[face].mean(0)))


def pck(pred_json, gt_json, h, thr=0.05):
    def kp(p):
        k = np.array(json.load(open(p))["people"][0]["pose_keypoints_2d"], np.float32).reshape(-1, 3)
        return k[BODY25_IDX if len(k) == 25 else COCO18_IDX]
    a, b = kp(pred_json), kp(gt_json)
    ok = (a[:, 2] > 0.1) & (b[:, 2] > 0.1)
    if not ok.any():
        return None
    return float((np.linalg.norm(a[ok, :2] - b[ok, :2], axis=1) < thr * h).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data_root", default="./data/VITON-HD")
    ap.add_argument("--split", default="test")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--pose_json", default=None)
    args = ap.parse_args()
    root = os.path.join(args.data_root, args.split)
    preds = sorted(p for p in glob.glob(os.path.join(args.pred, "*.png")) if not p.endswith(("_warp.png", "_seg.png")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}

    if args.paired:
        import lpips
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        lp = lpips.LPIPS(net="alex").to(dev)
        ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(dev)
        s_ssim, s_lp, s_glp = [], [], []
        for p in preds:
            name = os.path.basename(p).split("__")[0] + ".jpg"
            x = load(p)
            gt = load(os.path.join(root, "image", name), x.shape[:2])
            from PIL import Image
            seg = np.array(Image.open(os.path.join(root, "image-parse-v3", name.replace(".jpg", ".png"))))
            seg = group_parse(cv2.resize(seg, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_NEAREST))
            xt, gtt = to_t(x).to(dev), to_t(gt).to(dev)
            s_ssim.append(ssim(xt, gtt).item())
            s_lp.append(lp(xt, gtt).item())
            m = torch.from_numpy(seg == SEG_UPPER).float().to(dev)[None, None]
            s_glp.append(lp(xt * m, gtt * m).item())  # garment-region LPIPS
        res.update(SSIM=np.mean(s_ssim), LPIPS=np.mean(s_lp), LPIPS_garment=np.mean(s_glp))
    else:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        fid = FrechetInceptionDistance(normalize=True).to(dev)
        kid = KernelInceptionDistance(normalize=True, subset_size=min(1000, len(preds))).to(dev)
        reals = sorted(glob.glob(os.path.join(root, "image", "*.jpg")))
        hw = load(preds[0]).shape[:2]
        for paths, real in ((reals, True), (preds, False)):
            for i in range(0, len(paths), 32):
                batch = torch.cat([(to_t(load(q, hw)) + 1) / 2 for q in paths[i:i + 32]]).to(dev)
                fid.update(batch, real=real)
                kid.update(batch, real=real)
        res["FID"] = fid.compute().item()
        res["KID"] = kid.compute()[0].item()
        des = []
        for p in preds:
            sp = p[:-4] + "_seg.png"
            if os.path.exists(sp):
                seg = cv2.imread(sp, cv2.IMREAD_GRAYSCALE) // 30
                d = skin_delta_e(load(p), seg)
                if d is not None:
                    des.append(d)
        if des:
            res["Skin_dE"] = np.mean(des)

    if args.pose_json:
        scores = []
        for p in preds:
            stem = os.path.basename(p)[:-4]
            pj = os.path.join(args.pose_json, stem + "_keypoints.json")
            gj = os.path.join(root, "openpose_json", stem.split("__")[0] + "_keypoints.json")
            if os.path.exists(pj):
                s = pck(pj, gj, load(p).shape[0])
                if s is not None:
                    scores.append(s)
        res["PCK@0.05"] = np.mean(scores) if scores else float("nan")

    for k, v in res.items():
        print(f"{k:>14}: {v:.4f}")


if __name__ == "__main__":
    main()
