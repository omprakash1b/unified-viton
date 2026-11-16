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


# KGI segment groups (codes_tps/generate_tps.py). Neighbouring groups share boundary landmarks, so the
# separately fitted pieces agree on those points and meet without gaps.
KGI_GROUPS = [
    [4, 3, 2, 1, 0, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 31],  # torso
    [5, 6, 7, 12, 13, 14],                                        # left sleeve, upper
    [7, 8, 9, 10, 11, 12],                                        # left sleeve, lower
    [22, 23, 24, 29, 30, 31],                                     # right sleeve, upper
    [24, 25, 26, 27, 28, 29],                                     # right sleeve, lower
]
def polygon_mask(poly, h, w):
    """Rasterise closed polygons. poly [B,K,2] in [-1,1] -> [B,1,H,W] (even-odd rule, vectorised)."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=poly.device), torch.linspace(-1, 1, w, device=poly.device),
                            indexing="ij")
    px, py = xs.reshape(1, -1, 1), ys.reshape(1, -1, 1)               # [1,HW,1]
    x1, y1 = poly[..., 0][:, None], poly[..., 1][:, None]              # [B,1,K]
    x2, y2 = torch.roll(x1, -1, -1), torch.roll(y1, -1, -1)
    crosses = ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1)
    inside = crosses.sum(-1) % 2 == 1                                   # [B,HW]
    return inside.reshape(-1, 1, h, w).float()


def _min_std(pts, v):
    """Smallest standard deviation of the valid points along any direction (0 if collinear or duplicates)."""
    wgt = v.float()[..., None]
    n = wgt.sum(1).clamp(min=1)
    mu = (pts * wgt).sum(1) / n
    d = (pts - mu[:, None]) * wgt
    a, c, b = (d[..., 0] ** 2).sum(1) / n[:, 0], (d[..., 1] ** 2).sum(1) / n[:, 0], (d[..., 0] * d[..., 1]).sum(1) / n[:, 0]
    lam_min = (a + c) / 2 - torch.sqrt(((a - c) / 2) ** 2 + b ** 2)
    return lam_min.clamp(min=0).sqrt()


def _warp_part(cloth, region, grid):
    part = torch.cat([cloth * region, region], 1)
    w = F.grid_sample(part, grid.to(cloth.dtype), align_corners=True, padding_mode="zeros")
    return w[:, :3], w[:, 3:4]


def segment_tps_warp(cloth, cloth_mask, kc, kc_hat, min_std=0.01):
    """KGI-style segment-wise TPS. cloth [B,3,H,W] in [-1,1]; keypoints normalised [B,32,2].
    1) the whole garment is warped with one TPS through all keypoints (base layer, no holes);
    2) each KGI segment is cut out by the polygon of its landmarks (as KGI does), warped with its own TPS
       and pasted on top, torso first, then sleeves.
    A segment whose landmarks are collinear or duplicated (e.g. the sleeves of a vest, where KGI maps all
    sleeve points onto two landmarks) cannot define a 2-D warp: it is skipped and the base layer shows.
    Returns the premultiplied warp and its mask."""
    b, _, h, w = cloth.shape
    n = kc.shape[1]
    valid = (kc > -1.5).all(-1) & (kc_hat > -1.5).all(-1)
    out, out_mask = _warp_part(cloth, cloth_mask, tps_grid(kc, kc_hat, valid, h, w))  # base layer
    if n != 32:  # unknown landmark layout: base layer only
        return out, out_mask
    for idx in KGI_GROUPS:
        sel = torch.zeros(n, dtype=torch.bool, device=kc.device)
        sel[idx] = True
        v = valid & sel
        ok = (v[:, idx].all(1) & (_min_std(kc, v) > min_std) & (_min_std(kc_hat, v) > min_std))
        ok = ok.to(cloth.dtype)[:, None, None, None]
        region = polygon_mask(kc[:, idx], h, w) * cloth_mask
        grid = tps_grid(kc, kc_hat, torch.where(ok[:, 0, 0] > 0, v, valid), h, w)
        part, m = _warp_part(cloth, region, grid)
        m = m * ok
        out = out * (1 - m) + part * ok
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
    def __init__(self, n_pose=10, coarse_grad_scale=0.1, hybrid=True, feather=5):
        super().__init__()
        self.refiner = FlowRefiner(n_pose=n_pose, coarse_grad_scale=coarse_grad_scale)
        self.hybrid = hybrid    # output TPS texture + refined fill (see forward); False = refined warp only
        self.feather = feather  # odd kernel size, pixels

    def forward(self, cloth, cloth_mask, kc, kc_hat, seg_garment, occ, densepose, pose_maps):
        tps_c, tps_m = segment_tps_warp(cloth, cloth_mask, kc, kc_hat)
        # amodal target shape: visible garment + the part hidden behind the arms, so the flow lays the garment
        # out continuously behind an arm instead of squeezing it around the arm; the hidden part is cut below
        amodal = (seg_garment + occ).clamp(0, 1)
        coarse, flow = self.refiner(tps_c, tps_m, amodal, densepose, pose_maps)
        warped = apply_flow(torch.cat([tps_c, tps_m], 1), flow)
        keep = seg_garment * (1 - occ)  # clip to predicted garment region, cut where arms are in front
        r_mask = warped[:, 3:4] * keep
        r_cloth = warped[:, :3] * keep + (1 - r_mask)  # rgb is premultiplied; white outside, like worn_cloth
        out = {"refined": r_cloth, "refined_mask": r_mask, "tps": tps_c, "coarse_flow": coarse, "flow": flow}
        if not self.hybrid:
            out["warped"], out["warped_mask"] = r_cloth, r_mask
            return out
        # hybrid output: TPS texture (sharp logos/prints) wherever TPS covers the garment, the refined warp only
        # where TPS leaves holes. The TPS alpha is eroded and feathered by a few pixels so the two meet without
        # a visible line. The refiner is still trained on its own output ("refined"), see train.py.
        t_a = tps_m * keep
        k = self.feather
        t_a = -F.max_pool2d(-t_a, k, 1, k // 2)                       # erode
        t_a = F.avg_pool2d(t_a, k, 1, k // 2, count_include_pad=False)  # feather
        tps_rgb = tps_c / tps_m.clamp(min=1e-3)
        ref_rgb = warped[:, :3] / warped[:, 3:4].clamp(min=1e-3)
        has_ref = (r_mask > 1e-3).to(t_a.dtype)
        a = torch.where(has_ref > 0, t_a, (tps_m * keep > 1e-3).to(t_a.dtype))  # no refined pixel -> use TPS
        rgb = a * tps_rgb + (1 - a) * ref_rgb
        h_mask = torch.maximum(r_mask, tps_m * keep)
        out["warped"] = (rgb * h_mask + (1 - h_mask)).clamp(-1, 1)
        out["warped_mask"] = h_mask
        return out


def tv_loss(flow):
    return (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean() + (flow[..., 1:] - flow[..., :-1]).abs().mean()


def second_order_loss(flow):
    dy = flow[..., 2:, :] - 2 * flow[..., 1:-1, :] + flow[..., :-2, :]
    dx = flow[..., 2:] - 2 * flow[..., 1:-1] + flow[..., :-2]
    return dy.abs().mean() + dx.abs().mean()
