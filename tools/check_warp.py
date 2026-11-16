"""Check Stage 3 (segment TPS + flow refiner): numbers + overlay images.

    python tools/check_warp.py --split test --n 300 --save 20

Runs Stage 1 -> 2 -> 3 with checkpoints kp.pt, seg.pt, warp.pt on paired data and compares the warped
garment with the garment the person actually wears (worn_cloth), for two versions of the warp:
  TPS      = segment TPS only (no flow refiner), cut to the same predicted garment region
  refined  = full Stage 3 output (TPS + flow refiner)
  hybrid   = the Stage 3 output: TPS texture wherever TPS covers the garment, refined warp in the TPS holes,
             feathered seam (folds are not defined for it and print as nan)
Numbers (lower L1 / higher SSIM, IoU is better):
  L1      mean |warped - GT| inside the GT garment region (images in [-1, 1])
  SSIM    on the whole image (both have a white background)
  IoU     warped garment mask vs GT garment mask
  holes   share of GT garment pixels the warp leaves empty
  spill   share of warped garment pixels that fall outside the GT garment
  folds   share of garment pixels where the flow folds over itself (negative Jacobian) -> smeared texture
Hard-pose images (occluded / garment area >= --thr) are scored separately.
Images (results/check_warp/): cloth | TPS | refined | hybrid | GT worn garment | hybrid on person | flow magnitude
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vitonhd import SEG_UPPER, VitonHD  # noqa: E402
from models.warp import segment_tps_warp  # noqa: E402
from pipeline import Geometry, load_cfg, to_device  # noqa: E402

KEYS = ["L1", "SSIM", "IoU", "holes", "spill", "folds"]


def to_img(t):
    return cv2.cvtColor(((t.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def fold_rate(flow, region):
    """Share of pixels in `region` where the sampling grid (identity + flow) has a negative Jacobian."""
    b, _, h, w = flow.shape
    gx = flow[:, 0] + torch.linspace(-1, 1, w, device=flow.device).view(1, 1, w)
    gy = flow[:, 1] + torch.linspace(-1, 1, h, device=flow.device).view(1, h, 1)
    dxdx, dydx = gx[:, :-1, 1:] - gx[:, :-1, :-1], gy[:, :-1, 1:] - gy[:, :-1, :-1]
    dxdy, dydy = gx[:, 1:, :-1] - gx[:, :-1, :-1], gy[:, 1:, :-1] - gy[:, :-1, :-1]
    det = dxdx * dydy - dydx * dxdy
    r = region[:, 0, :-1, :-1] > 0.5
    return ((det < 0) & r).flatten(1).sum(1).float() / r.flatten(1).sum(1).clamp(min=1)


def scores(warped, w_mask, gt, gt_m):
    """Per-image metrics; all inputs [B,C,H,W]."""
    pm, gm = w_mask > 0.5, gt_m > 0.5
    l1 = ((warped - gt).abs().mean(1, keepdim=True) * gt_m).flatten(1).sum(1) / gt_m.flatten(1).sum(1).clamp(min=1)
    s = ssim(((warped + 1) / 2).float(), ((gt + 1) / 2).float(), data_range=1.0, reduction="none")
    inter, union = (pm & gm).flatten(1).sum(1).float(), (pm | gm).flatten(1).sum(1).float()
    holes = (gm & ~pm).flatten(1).sum(1).float() / gm.flatten(1).sum(1).clamp(min=1)
    spill = (pm & ~gm).flatten(1).sum(1).float() / pm.flatten(1).sum(1).clamp(min=1)
    return {"L1": l1, "SSIM": s, "IoU": inter / union.clamp(min=1), "holes": holes, "spill": spill}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300, help="samples to score (0 = all)")
    ap.add_argument("--save", type=int, default=20, help="overlay images to write")
    ap.add_argument("--thr", type=float, default=0.08, help="hard-pose threshold: occluded / garment area")
    ap.add_argument("--out", default="results/check_warp")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geo = Geometry(cfg).to(dev).eval()
    geo.load(cfg, "gen")  # kp.pt, seg.pt, warp.pt

    ds = VitonHD(cfg, args.split, paired=True)
    if args.n:
        ds = Subset(ds, np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int).tolist())
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    os.makedirs(args.out, exist_ok=True)

    NAMES = ("TPS", "refined", "hybrid")
    res = {k: {n: {m: [] for m in KEYS} for n in NAMES} for k in ("all", "hard")}
    saved = 0
    with torch.no_grad():
        for b in dl:
            b = to_device(b, dev)
            g = geo(b)
            gt, gt_m = b["worn_cloth"], (b["seg"] == SEG_UPPER).float()[:, None]
            occ_gt = b["occ"]
            hard = (occ_gt.flatten(1).sum(1) / (gt_m + occ_gt).flatten(1).sum(1).clamp(min=1)) >= args.thr

            keep = g["seg_soft"][:, SEG_UPPER:SEG_UPPER + 1] * (1 - g["occ"])  # same cut as the refined warp
            tps_c, tps_mask = segment_tps_warp(b["cloth"], b["cloth_mask"], g["kc"], g["kc_hat"])
            tps_cut_m = tps_mask * keep
            tps_cut = tps_c * keep + (1 - tps_cut_m)  # TPS output is premultiplied; white outside

            hyb, hyb_m = g["warped"], g["warped_mask"]  # Stage 3 output: TPS texture + refined fill

            region = (gt_m + g["refined_mask"]).clamp(0, 1)
            for name, (wc, wm, fl) in {"TPS": (tps_cut, tps_cut_m, "zero"),
                                       "refined": (g["refined"], g["refined_mask"], g["flow"]),
                                       "hybrid": (hyb, hyb_m, "nan")}.items():
                sc = scores(wc, wm, gt, gt_m)
                if isinstance(fl, str):
                    sc["folds"] = torch.full((len(gt),), 0.0 if fl == "zero" else float("nan"), device=dev)
                else:
                    sc["folds"] = fold_rate(fl.float(), region)
                for k in KEYS:
                    v = sc[k].float().cpu()
                    res["all"][name][k].append(v)
                    res["hard"][name][k].append(v[hard.cpu()])

            for i in range(len(gt)):
                if saved >= args.save:
                    break
                mag = g["flow"][i].float().norm(dim=0).cpu().numpy()
                mag = cv2.applyColorMap((mag / max(mag.max(), 1e-6) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                wm = hyb_m[i]
                on_person = to_img(b["agnostic"][i] * (1 - wm) + (hyb[i] - (1 - wm)))
                panels = [to_img(b["cloth"][i]), to_img(tps_cut[i]), to_img(g["refined"][i]), to_img(hyb[i]),
                          to_img(gt[i]), on_person, mag]
                cv2.imwrite(os.path.join(args.out, b["name"][i].replace(".jpg", "_warp.png")), np.hstack(panels))
                saved += 1

    for subset, title in (("all", f"{args.split}: all"), ("hard", f"{args.split}: hard pose (occlusion >= {args.thr:.2f})")):
        n = sum(len(v) for v in res[subset]["refined"]["L1"])
        if n == 0:
            continue
        print(f"\n[{title}]  images={n}")
        print("            " + "".join(f"{k:>9s}" for k in KEYS))
        for name in NAMES:
            vals = [torch.cat(res[subset][name][k]).mean().item() for k in KEYS]
            print(f"  {name:9s} " + "".join(f"{v:9.4f}" for v in vals))
    print(f"\noverlays: {saved} images in {args.out}")


if __name__ == "__main__":
    main()