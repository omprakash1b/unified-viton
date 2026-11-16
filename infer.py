"""Full-pipeline inference.

    python infer.py --split test --unpaired --out results/unpaired
Saves try-on images plus (optionally) the intermediate warped garment and parsing.
"""
import argparse
import os

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vitonhd import VitonHD
from models.emasc import SkipDecoder, composite
from models.generator import TryOnGenerator, new_skin_mask
from pipeline import Geometry, build_conditions, ckpt, load_cfg, load_sd, to_device


def save_img(t, path):
    arr = ((t.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
    Image.fromarray(arr).save(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--unpaired", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--save_intermediate", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    dev = "cuda"
    dt = torch.float16 if cfg["train"]["mixed_precision"] == "fp16" else torch.bfloat16
    os.makedirs(args.out, exist_ok=True)

    geo = Geometry(cfg).load(cfg, "gen").to(dev).eval()
    sd = load_sd(cfg, dev, dt)
    gen = TryOnGenerator(sd["unet"], cfg["model"]["num_seg_classes"], cfg["data"]["num_body_kpts"],
                         n_garment_tokens=cfg["model"]["num_garment_tokens"],
                         n_skin_tokens=cfg["model"]["num_skin_tokens"],
                         aspect=cfg["data"]["height"] / cfg["data"]["width"])
    gen.load_state_dict(torch.load(ckpt(cfg, "generator"), map_location="cpu"))
    gen.to(dev, dt).eval()
    dec = SkipDecoder(sd["vae"].float()).to(dev)  # decode in fp32 for clean detail
    dec.load_state_dict(torch.load(ckpt(cfg, "decoder"), map_location="cpu"), strict=False)

    sched = DDIMScheduler.from_config(sd["sched"].config)
    sched.set_timesteps(cfg["infer"]["steps"], device=dev)
    gscale = cfg["infer"]["guidance"]

    ds = VitonHD(cfg, args.split, paired=not args.unpaired)
    for b in tqdm(DataLoader(ds, batch_size=args.batch, num_workers=4)):
        b = to_device(b, dev)
        g = geo(b)
        c = build_conditions(sd, b, g)
        skin_mask = new_skin_mask(g["seg_soft"], b["mask_a"])
        n = b["person"].shape[0]
        text = sd["text_emb"].expand(n, -1, -1)
        z = torch.randn(n, 4, *c["agn_lat"].shape[-2:], device=dev, dtype=dt)
        for t in sched.timesteps:
            x_in = gen.build_input(z, c["mask_lat"].to(dt), c["agn_lat"].to(dt), c["warp_lat"].to(dt), c["seg_lat"].to(dt))
            tt = t.expand(n)
            eps_c = gen(x_in, tt, text, c["garment_clip"], c["skin_clip"], skin_mask.to(dt), c["control"].to(dt))
            if gscale != 1.0:  # classifier-free guidance over the garment / skin tokens
                eps_u = gen(x_in, tt, text, None, None, None, c["control"].to(dt))
                eps = eps_u + gscale * (eps_c - eps_u)
            else:
                eps = eps_c
            z = sched.step(eps, t, z).prev_sample
        img = dec(z.float(), b["agnostic"], g["warped"], b["mask_a"], g["warped_mask"])
        img = composite(img, b["person"], b["mask_a"])
        for i in range(n):
            stem = f"{os.path.splitext(b['name'][i])[0]}__{os.path.splitext(b['cloth_name'][i])[0]}"
            save_img(img[i], os.path.join(args.out, stem + ".png"))
            if args.save_intermediate:
                save_img(g["warped"][i], os.path.join(args.out, stem + "_warp.png"))
                seg = g["seg_soft"][i].argmax(0).byte().cpu().numpy() * 30
                Image.fromarray(seg).save(os.path.join(args.out, stem + "_seg.png"))


if __name__ == "__main__":
    main()
