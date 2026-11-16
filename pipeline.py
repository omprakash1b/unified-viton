"""Glue: model loading, frozen upstream stages, conditioning tensors."""
import os

import torch
import torch.nn.functional as F
import yaml

from models.keypoint_gcn import KeypointGCN, denormalise_xy, normalise_xy
from models.seg_net import SegNet
from models.warp import GarmentWarper

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ckpt(cfg, name):
    return os.path.join(cfg["train"]["out_dir"], f"{name}.pt")


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def render_kpt_maps(xy_px, h, w, sigma):
    """xy_px: [B,N,2] pixel coords (negative = missing) -> [B,N,H,W] gaussians (differentiable)."""
    ys = torch.arange(h, device=xy_px.device, dtype=xy_px.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=xy_px.device, dtype=xy_px.dtype).view(1, 1, 1, w)
    x, y = xy_px[..., 0, None, None], xy_px[..., 1, None, None]
    m = torch.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return m * (xy_px[..., :1, None] >= 0).to(m.dtype)


# ---------------------------------------------------------------------------
# Geometry stages S1-S3
# ---------------------------------------------------------------------------
class Geometry(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, m = cfg["data"], cfg["model"]
        self.h, self.w = d["height"], d["width"]
        self.sigma = d["kpt_sigma"] * self.h / 512
        self.kp = KeypointGCN(d["num_garment_kpts"], d["num_body_kpts"])
        self.seg = SegNet(m["num_seg_classes"], d["num_garment_kpts"], d["num_body_kpts"])
        self.warp = GarmentWarper(d["num_body_kpts"], cfg["stages"]["warp"]["coarse_grad_scale"])

    def load(self, cfg, upto):
        """Load trained weights for stages before `upto` ('seg', 'warp', 'gen')."""
        order = ["kp", "seg", "warp"]
        for name in order[: {"seg": 1, "warp": 2, "gen": 3}[upto]]:
            getattr(self, name).load_state_dict(torch.load(ckpt(cfg, name), map_location="cpu"))
        return self

    def pose_input(self, b):
        pose = b["pose"].clone()
        pose[..., :2] = normalise_xy(b["pose"][..., :2], self.h, self.w)
        return pose

    def run_kp(self, b):
        kc = normalise_xy(b["kc"], self.h, self.w)
        kc_hat = self.kp(kc, self.pose_input(b), b["vis"], b["depth"])
        kc_hat = torch.where(kc < -1.5, kc, kc_hat)  # keep missing points missing
        return kc, kc_hat

    def run_seg(self, b, kc_hat):
        px = torch.where(kc_hat < -1.5, torch.full_like(kc_hat, -1.0), denormalise_xy(kc_hat, self.h, self.w))
        maps = render_kpt_maps(px, self.h, self.w, self.sigma)
        seg_logits, occ_logit = self.seg(b["seg_agnostic"], maps, b["pose_maps"], b["densepose"], b["cloth_mask"])
        return seg_logits, occ_logit

    def forward(self, b):
        kc, kc_hat = self.run_kp(b)
        seg_logits, occ_logit = self.run_seg(b, kc_hat)
        seg_soft = seg_logits.softmax(1)
        occ = torch.sigmoid(occ_logit)
        w = self.warp(b["cloth"], b["cloth_mask"], kc, kc_hat, seg_soft[:, 3:4], occ, b["densepose"], b["pose_maps"])
        return {"kc": kc, "kc_hat": kc_hat, "seg_logits": seg_logits, "seg_soft": seg_soft, "occ": occ, **w}


# ---------------------------------------------------------------------------
# Stable Diffusion components
# ---------------------------------------------------------------------------
def load_sd(cfg, device, dtype):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

    repo = cfg["model"]["sd_inpaint"]
    tok = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder").to(device, dtype).requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae").to(device, dtype).requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet")
    sched = DDPMScheduler.from_pretrained(repo, subfolder="scheduler")
    clip = CLIPVisionModel.from_pretrained(cfg["model"]["clip_vision"]).to(device, dtype).requires_grad_(False)
    with torch.no_grad():
        ids = tok(cfg["model"]["prompt"], padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids
        text_emb = te(ids.to(device))[0]
    return {"tok": tok, "vae": vae, "unet": unet, "sched": sched, "clip": clip, "text_emb": text_emb}


@torch.no_grad()
def clip_features(clip, img):
    """img [-1,1] -> CLIP patch features [B,257,1024]."""
    x = F.interpolate((img + 1) / 2, size=(224, 224), mode="bicubic", align_corners=False)
    x = (x - CLIP_MEAN.to(x)) / CLIP_STD.to(x)
    return clip(pixel_values=x.to(next(clip.parameters()).dtype)).last_hidden_state


@torch.no_grad()
def encode(vae, img):
    return vae.encode(img.to(vae.dtype)).latent_dist.sample() * vae.config.scaling_factor


@torch.no_grad()
def build_conditions(sd, b, geo):
    """All tensors the generator needs, from a batch + geometry outputs."""
    vae = sd["vae"]
    lat_hw = (b["person"].shape[-2] // 8, b["person"].shape[-1] // 8)
    return {
        "agn_lat": encode(vae, b["agnostic"]),
        "warp_lat": encode(vae, geo["warped"]),
        "mask_lat": F.interpolate(b["mask_a"], size=lat_hw, mode="nearest"),
        "seg_lat": F.interpolate(geo["seg_soft"], size=lat_hw, mode="bilinear", align_corners=False),
        "control": torch.cat([b["pose_maps"], b["densepose"]], 1),
        "garment_clip": clip_features(sd["clip"], b["cloth"]),
        "skin_clip": clip_features(sd["clip"], b["skin_ref"]),
    }
