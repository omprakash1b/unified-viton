"""Stage 3 — two-step garment warping.

1. Segment-wise TPS (KGI): the garment is split into five sub-segments
   (centre, left-up, left-low, right-up, right-low); each gets its own TPS K_c -> K̂_c.
2. Flow refinement (SD-VITON): coarse flow layers are trained TV-dominant, a final
   refinement layer sees all losses (appearance gradients into the coarse layers are scaled down).
3. Occlusion/layout masking (HR-VITON style): Î_c = W(I_c, F∘T) ⊙ Ŝ_garment ⊙ (1 − O)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import UNet

SEG_ORDER = [0, 2, 4, 1, 3]  # composite order: centre, low sleeves, up sleeves on top


# ---------------------------------------------------------------------------
# Thin-plate spline (batched, differentiable)
# ---------------------------------------------------------------------------
def _tps_kernel(d2):
    return d2 * torch.log(d2.clamp(min=1e-9))


def tps_grid(src, dst, valid, h, w, reg=1e-3):
    """Backward-warp grid: for every output pixel (person space) the sampling location in the
    garment image. src/dst: [B,N,2] in [-1,1]; valid: [B,N] bool. Returns [B,H,W,2]."""
    b, n, _ = dst.shape
    dev, dt = dst.device, torch.float32
    src, dst = src.to(dt), dst.to(dt)
    d2 = ((dst[:, :, None] - dst[:, None]) ** 2).sum(-1)
    lam = torch.where(valid, torch.full_like(d2[..., 0], reg), torch.full_like(d2[..., 0], 1e4))
    k = _tps_kernel(d2) + torch.diag_embed(lam)
    p = torch.cat([torch.ones(b, n, 1, device=dev), dst], -1)
    top = torch.cat([k, p], -1)
    bot = torch.cat([p.transpose(1, 2), torch.zeros(b, 3, 3, device=dev)], -1)
    lhs = torch.cat([top, bot], 1)
    rhs = torch.cat([src, torch.zeros(b, 3, 2, device=dev)], 1)
    params = torch.linalg.solve(lhs, rhs)  # [B,N+3,2]
    wts, aff = params[:, :n], params[:, n:]
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=dev), torch.linspace(-1, 1, w, device=dev), indexing="ij")
    g = torch.stack([xs, ys], -1).reshape(1, -1, 2).expand(b, -1, -1)
    u = _tps_kernel(((g[:, :, None] - dst[:, None]) ** 2).sum(-1))  # [B,HW,N]
    out = aff[:, :1] + g @ aff[:, 1:] + u @ wts
    return out.reshape(b, h, w, 2)


# ---------------------------------------------------------------------------
# Segment assignment
# ---------------------------------------------------------------------------
def assign_segments(kc, valid, centre_frac=0.2):
    """Geometric 5-way split of in-shop keypoints (robust to landmark ordering).
    Returns [B,N] long in {0:centre, 1:left-up, 2:left-low, 3:right-up, 4:right-low}."""
    x, y = kc[..., 0], kc[..., 1]
    big = torch.full_like(x, 1e4)
    xmin = torch.where(valid, x, big).min(1, keepdim=True).values
    xmax = torch.where(valid, x, -big).max(1, keepdim=True).values
    cx, span = (xmin + xmax) / 2, (xmax - xmin).clamp(min=1e-3)
    ymed = torch.where(valid, y, torch.nan).nanmedian(1, keepdim=True).values
    left, up = x < cx, y < ymed
    seg = torch.where(left, torch.where(up, 1, 2), torch.where(up, 3, 4))
    return torch.where((x - cx).abs() < centre_frac * span, torch.zeros_like(seg), seg)


def pixel_segments(kc, seg_ids, valid, cloth_mask):
    """Label each garment pixel with the segment of its nearest keypoint -> [B,5,H,W] masks."""
    b, _, h, w = cloth_mask.shape
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=kc.device), torch.linspace(-1, 1, w, device=kc.device), indexing="ij")
    g = torch.stack([xs, ys], -1).reshape(1, -1, 1, 2)
    d = ((g - kc[:, None]) ** 2).sum(-1)  # [B,HW,N]
    d = torch.where(valid[:, None], d, torch.full_like(d, 1e4))
    nearest = d.argmin(-1)  # [B,HW]
    lab = torch.gather(seg_ids, 1, nearest).reshape(b, 1, h, w)
    return F.one_hot(lab[:, 0], 5).permute(0, 3, 1, 2).float() * cloth_mask


def segment_tps_warp(cloth, cloth_mask, kc, kc_hat):
    """KGI-style segment-wise TPS. cloth [B,3,H,W] in [-1,1]; keypoints normalised."""
    b, _, h, w = cloth.shape
    valid = (kc > -1.5).all(-1) & (kc_hat > -1.5).all(-1)
    seg_ids = assign_segments(kc, valid)
    seg_masks = pixel_segments(kc, seg_ids, valid, cloth_mask)
    out = torch.zeros_like(cloth)
    out_mask = torch.zeros_like(cloth_mask)
    for s in SEG_ORDER:
        v = valid & (seg_ids == s)
        few = v.sum(1) < 4  # too few points -> fall back to all keypoints
        v = torch.where(few[:, None], valid, v)
        grid = tps_grid(kc, kc_hat, v, h, w)
        part = torch.cat([cloth * seg_masks[:, s:s + 1], seg_masks[:, s:s + 1]], 1)
        warped = F.grid_sample(part, grid.to(cloth.dtype), align_corners=True, padding_mode="zeros")
        m = warped[:, 3:4]
        out = out * (1 - m) + warped[:, :3]
        out_mask = torch.maximum(out_mask, m)
    return out, out_mask


# ---------------------------------------------------------------------------
# Flow refinement with split objectives (SD-VITON)
# ---------------------------------------------------------------------------
class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


class FlowRefiner(nn.Module):
    def __init__(self, n_pose=10, base=48, max_flow=0.15, coarse_grad_scale=0.1):
        super().__init__()
        cin = 3 + 1 + 1 + 3 + n_pose  # TPS garment, TPS mask, Ŝ_garment, DensePose, pose maps
        self.body = UNet(cin, 2, base=base, return_features=True)  # its 1x1 head = coarse flow
        self.refine = nn.Sequential(nn.Conv2d(base + 2, base, 3, padding=1), nn.SiLU(), nn.Conv2d(base, 2, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        self.max_flow, self.s = max_flow, coarse_grad_scale

    def forward(self, tps_cloth, tps_mask, seg_garment, densepose, pose_maps):
        x = torch.cat([tps_cloth, tps_mask, seg_garment, densepose, pose_maps], 1)
        coarse_raw, feat = self.body(x)
        coarse = self.max_flow * torch.tanh(coarse_raw)
        # appearance losses reach the coarse (TV-dominant) layers only through a scaled gradient
        c_s, f_s = _GradScale.apply(coarse, self.s), _GradScale.apply(feat, self.s)
        final = c_s + self.max_flow * torch.tanh(self.refine(torch.cat([f_s, c_s], 1)))
        return coarse, final  # [B,2,H,W] each, normalised offsets


def apply_flow(img, flow):
    b, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=img.device), torch.linspace(-1, 1, w, device=img.device), indexing="ij")
    base = torch.stack([xs, ys], -1)[None].to(img.dtype)
    return F.grid_sample(img, base + flow.permute(0, 2, 3, 1), align_corners=True, padding_mode="zeros")


class GarmentWarper(nn.Module):
    def __init__(self, n_pose=10, coarse_grad_scale=0.1):
        super().__init__()
        self.refiner = FlowRefiner(n_pose=n_pose, coarse_grad_scale=coarse_grad_scale)

    def forward(self, cloth, cloth_mask, kc, kc_hat, seg_garment, occ, densepose, pose_maps):
        tps_c, tps_m = segment_tps_warp(cloth, cloth_mask, kc, kc_hat)
        coarse, flow = self.refiner(tps_c, tps_m, seg_garment, densepose, pose_maps)
        warped = apply_flow(torch.cat([tps_c, tps_m], 1), flow)
        keep = seg_garment * (1 - occ)  # clip to predicted garment region, cut where arms are in front
        w_mask = warped[:, 3:4] * keep
        w_cloth = warped[:, :3] * keep + (1 - w_mask)  # rgb is premultiplied; white outside, like worn_cloth
        return {"warped": w_cloth, "warped_mask": w_mask, "tps": tps_c, "coarse_flow": coarse, "flow": flow}


def tv_loss(flow):
    return (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean() + (flow[..., 1:] - flow[..., :-1]).abs().mean()


def second_order_loss(flow):
    dy = flow[..., 2:, :] - 2 * flow[..., 1:-1, :] + flow[..., :-2, :]
    dx = flow[..., 2:] - 2 * flow[..., 1:-1] + flow[..., :-2]
    return dy.abs().mean() + dx.abs().mean()
