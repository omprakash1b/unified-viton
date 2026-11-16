"""Stage-wise training (doc section "Training strategy").

    python train.py --stage kp        # T1  keypoint GCN
    python train.py --stage seg       # T2  segmentation + occlusion
    python train.py --stage warp      # T3  segment TPS + flow refiner
    python train.py --stage unet      # T4a widened UNet + garment tokens
    python train.py --stage branches  # T4b pose ControlNet + skin attention (UNet frozen)
    python train.py --stage decoder   # T5  skip-connection decoder
Each stage freezes the ones before it and trains on their *predicted* outputs.
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vitonhd import SEG_UPPER, VitonHD
from models.keypoint_gcn import keypoint_loss, normalise_xy
from models.losses import VGGLoss, masked_l1
from models.seg_net import seg_loss
from models.warp import second_order_loss, tv_loss
from pipeline import Geometry, build_conditions, ckpt, load_cfg, load_sd, to_device


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def drop(x, p):
    return None if (x is None or random.random() < p) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stage", required=True, choices=["kp", "seg", "warp", "unet", "branches", "decoder"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (debug)")
    ap.add_argument("--batch", type=int, default=0, help="override the stage batch size (small GPUs)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--height", type=int, default=0, help="override training height")
    ap.add_argument("--width", type=int, default=0, help="override training width")
    args = ap.parse_args()

    if args.accum < 1:
        ap.error("--accum must be >= 1")
    if args.batch < 0:
        ap.error("--batch must be >= 0")
    if bool(args.height) != bool(args.width):
        ap.error("--height and --width must be supplied together")
    if args.height < 0 or args.width < 0:
        ap.error("--height and --width must be >= 0")

    cfg = load_cfg(args.config)
    sc = cfg["stages"][args.stage]
    if args.batch:
        sc["batch"] = args.batch
    if args.height and args.width:
        old_height = cfg["data"]["height"]
        cfg["data"]["height"], cfg["data"]["width"] = args.height, args.width
        cfg["data"]["kpt_sigma"] *= args.height / old_height
    seed_all(cfg["train"]["seed"])
    os.makedirs(cfg["train"]["out_dir"], exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    want_bf16 = cfg["train"]["mixed_precision"] == "bf16"
    if want_bf16 and dev == "cuda" and not torch.cuda.is_bf16_supported():
        print("bf16 not supported on this GPU -> using fp16 with loss scaling")
        want_bf16 = False
    amp = torch.bfloat16 if want_bf16 else torch.float16
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp == torch.float16)

    ds = VitonHD(cfg, "train", paired=True)
    dl = DataLoader(ds, batch_size=sc["batch"], shuffle=True, num_workers=cfg["train"]["num_workers"],
                    pin_memory=False, drop_last=True, persistent_workers=cfg["train"]["num_workers"] > 0)

    geo = Geometry(cfg).to(dev)
    sd = gen = dec = vgg = lp = None

    # ---------------- what trains in this stage ----------------
    if args.stage == "kp":
        trainable = geo.kp
    elif args.stage == "seg":
        geo.load(cfg, "seg")
        trainable = geo.seg
    elif args.stage == "warp":
        geo.load(cfg, "warp")
        trainable = geo.warp
        vgg = VGGLoss().to(dev)
    elif args.stage in ("unet", "branches"):
        from models.generator import TryOnGenerator
        geo.load(cfg, "gen")
        # Keep SD weights in fp32 for fp16 training; bf16 weights may use bf16.
        sd = load_sd(cfg, dev, amp if (use_amp and amp == torch.bfloat16) else torch.float32)
        gen = TryOnGenerator(sd["unet"], cfg["model"]["num_seg_classes"], cfg["data"]["num_body_kpts"],
                             n_garment_tokens=cfg["model"]["num_garment_tokens"],
                             n_skin_tokens=cfg["model"]["num_skin_tokens"],
                             aspect=cfg["data"]["height"] / cfg["data"]["width"]).to(dev)
        if args.stage == "branches":
            gen.load_state_dict(torch.load(ckpt(cfg, "generator_t4a"), map_location="cpu"))
            gen.reinit_controlnet()
            gen.requires_grad_(False)
            for p in gen.params_t4b():
                p.requires_grad_(True)
        gen.unet.enable_gradient_checkpointing()
        gen.controlnet.enable_gradient_checkpointing()
        trainable = gen
    else:  # decoder
        from models.emasc import SkipDecoder, composite
        geo.load(cfg, "gen")
        sd = load_sd(cfg, dev, torch.float32)
        dec = SkipDecoder(sd["vae"]).to(dev)
        trainable = dec
        try:
            import lpips
            lp = lpips.LPIPS(net="vgg").to(dev).requires_grad_(False)
        except ImportError:
            lp = VGGLoss().to(dev)

    geo.requires_grad_(False)
    if args.stage in ("kp", "seg", "warp"):
        trainable.requires_grad_(True)
    params = {
        "unet": lambda: gen.params_t4a(),
        "branches": lambda: gen.params_t4b(),
        "decoder": lambda: dec.parameters_trainable(),
    }.get(args.stage, lambda: list(trainable.parameters()))()
    params = [p for p in params if p.requires_grad]
    print(f"[{args.stage}] trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=sc["lr"], weight_decay=1e-2)

    save_name = {"unet": "generator_t4a", "branches": "generator", "decoder": "decoder"}.get(args.stage, args.stage)
    if args.resume and os.path.exists(ckpt(cfg, save_name)):
        trainable.load_state_dict(torch.load(ckpt(cfg, save_name), map_location="cpu"), strict=False)

    def save():
        state = trainable.state_dict()
        if args.stage == "decoder":  # adapters only, the VAE is the stock one
            state = {k: v for k, v in state.items() if k.startswith("adapt_")}
        torch.save(state, ckpt(cfg, save_name))

    print(f"[{args.stage}] batch {sc['batch']} x accum {args.accum} "
          f"at {cfg['data']['height']}x{cfg['data']['width']}")
    step = 0
    opt.zero_grad(set_to_none=True)

    def optimizer_update(loss_for_log, logs_for_log, pending):
        nonlocal step
        # Correct the scale when flushing an incomplete accumulation group.
        scaler.unscale_(opt)
        if pending < args.accum:
            correction = args.accum / pending
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg["train"]["log_every"] == 0:
            print(f"step {step} loss {loss_for_log.item():.4f} " +
                  " ".join(f"{k} {v:.4f}" for k, v in logs_for_log.items()))
        if args.max_steps and step >= args.max_steps:
            save()
            return True
        return False

    for epoch in range(sc["epochs"]):
        trainable.train()
        pending = 0
        last_loss, last_logs = None, {}
        for b in tqdm(dl, desc=f"{args.stage} ep{epoch}"):
            b = to_device(b, dev)
            with torch.autocast(dev, dtype=amp, enabled=use_amp):
                loss, logs = stage_loss(args.stage, cfg, sc, b, geo, sd, gen, dec, vgg, lp)
            scaler.scale(loss / args.accum).backward()
            pending += 1
            last_loss, last_logs = loss, logs

            if pending == args.accum:
                if optimizer_update(last_loss, last_logs, pending):
                    return
                pending = 0

        # Apply the remaining gradients rather than silently discarding samples.
        if pending:
            if optimizer_update(last_loss, last_logs, pending):
                return

        if (epoch + 1) % cfg["train"]["save_every_epochs"] == 0 or epoch + 1 == sc["epochs"]:
            save()


def stage_loss(stage, cfg, sc, b, geo, sd, gen, dec, vgg, lp):
    h, w = cfg["data"]["height"], cfg["data"]["width"]
    if stage == "kp":
        _, kc_hat = geo.run_kp(b)
        loss = keypoint_loss(kc_hat, normalise_xy(b["kc_gt"], h, w))
        return loss, {}

    if stage == "seg":
        with torch.no_grad():
            _, kc_hat = geo.run_kp(b)
        seg_logits, occ_logit = geo.run_seg(b, kc_hat)
        return seg_loss(seg_logits.float(), occ_logit.float(), b["seg"], b["occ"], cfg["model"]["num_seg_classes"],
                        sc["lambda_dice"], sc["lambda_occ"])

    if stage == "warp":
        with torch.no_grad():
            kc, kc_hat = geo.run_kp(b)
            seg_logits, occ_logit = geo.run_seg(b, kc_hat)
        out = geo.warp(b["cloth"], b["cloth_mask"], kc, kc_hat, seg_logits.softmax(1)[:, SEG_UPPER:SEG_UPPER + 1],
                       torch.sigmoid(occ_logit), b["densepose"], b["pose_maps"])
        gt_m = (b["seg"] == SEG_UPPER).float()[:, None]
        l1 = masked_l1(out["warped"].float(), b["worn_cloth"], gt_m) + F.l1_loss(out["warped_mask"].float(), gt_m)
        lv = vgg(out["warped"].float(), b["worn_cloth"])
        ltv = tv_loss(out["coarse_flow"].float())
        l2 = second_order_loss(out["flow"].float())
        loss = l1 + sc["lambda_vgg"] * lv + sc["lambda_tv"] * ltv + sc["lambda_2nd"] * l2
        return loss, {"l1": l1.item(), "vgg": lv.item(), "tv": ltv.item(), "2nd": l2.item()}

    if stage in ("unet", "branches"):
        from models.generator import new_skin_mask
        with torch.no_grad():
            g = geo(b)
            c = build_conditions(sd, b, g)
            z0 = sd["vae"].encode(b["person"].to(sd["vae"].dtype)).latent_dist.sample() * sd["vae"].config.scaling_factor
        sched = sd["sched"]
        noise = torch.randn_like(z0)
        t = torch.randint(0, sched.config.num_train_timesteps, (z0.shape[0],), device=z0.device)
        z_t = sched.add_noise(z0, noise, t)
        x_in = gen.build_input(z_t, c["mask_lat"].to(z_t), c["agn_lat"], c["warp_lat"], c["seg_lat"].to(z_t))
        p = sc["cond_drop"]
        text = sd["text_emb"].expand(z0.shape[0], -1, -1)
        if stage == "unet":  # T4a: base UNet + garment tokens only
            eps = gen(x_in, t, text, garment_clip=drop(c["garment_clip"], p))
        else:  # T4b: add pose ControlNet + skin branch
            skin_mask = new_skin_mask(g["seg_soft"], b["mask_a"])
            eps = gen(x_in, t, text, garment_clip=drop(c["garment_clip"], p), skin_clip=drop(c["skin_clip"], p),
                      skin_mask=skin_mask, control=drop(c["control"], p))
        l_diff = F.mse_loss(eps.float(), noise.float())
        a = sched.alphas_cumprod.to(z0.device)[t].view(-1, 1, 1, 1)
        z0_hat = (z_t.float() - (1 - a).sqrt() * eps.float()) / a.sqrt()
        gm = F.interpolate(g["warped_mask"].float(), size=z0.shape[-2:], mode="area")
        l_gp = ((z0_hat - z0.float()).abs() * gm).sum() / (gm.sum() * 4).clamp(min=1)
        return l_diff + sc["lambda_gp"] * l_gp, {"diff": l_diff.item(), "gp": l_gp.item()}

    if stage == "decoder":
        from models.emasc import composite
        with torch.no_grad():
            g = geo(b)
            z0 = sd["vae"].encode(b["person"]).latent_dist.mode() * sd["vae"].config.scaling_factor
        out = dec(z0, b["agnostic"], g["warped"], b["mask_a"], g["warped_mask"])
        img = composite(out, b["person"], b["mask_a"])
        l1 = F.l1_loss(img.float(), b["person"])
        lpv = lp(img.float(), b["person"]).mean()
        return l1 + sc["lambda_lpips"] * lpv, {"l1": l1.item(), "lpips": lpv.item()}
    raise ValueError(stage)


if __name__ == "__main__":
    main()
