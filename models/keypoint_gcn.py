"""Stage 1 — pose-aware garment keypoint prediction.

KGI's two-stream graph network (garment graph = main stream, pose graph = side stream that
conditions it at every level), extended so every pose node also carries SMPL/parsing-derived
visibility v and depth d:  K̂_c = G_kp(K_c, [K_p || v_p || d_p]).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# KGI pose graph: 10 upper-body joints, 9 bones (18 directed edges)
POSE_EDGES = [(1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (1, 9)]


# KGI's garment graph (codes_kg/train_kg.py). Landmark order: 0-4 neckline (2 = front centre),
# 5 left shoulder, 6-9 left sleeve outer to cuff, 10-13 left sleeve inner up, 14 left armpit, 15-17 left side
# to hem, 18 hem centre, 19-21 right side up, 22 right armpit, 23-26 right sleeve inner down, 27-30 right
# sleeve outer up, 31 right shoulder.
KGI_CONTOUR = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 31)] + [(i, i - 1) for i in range(31, 5, -1)] + [(5, 0)]
KGI_SYMMETRY = [(0, 4), (1, 3), (5, 31), (14, 22), (15, 21), (16, 20), (17, 19),   # left <-> right
                (6, 13), (7, 12), (8, 11), (23, 30), (24, 29), (25, 28),          # across each sleeve
                (2, 18)]                                                          # neck centre <-> hem centre


def garment_edges(n=32, sym_pairs=None):
    """KGI's 32 contour + 14 symmetry edges (64 + 28 directed). For another landmark set pass n != 32:
    a plain contour ring plus `sym_pairs`."""
    if n == 32:
        return KGI_CONTOUR + (KGI_SYMMETRY if sym_pairs is None else list(sym_pairs))
    return [(i, (i + 1) % n) for i in range(n)] + list(sym_pairs or [])


def adjacency(n, edges):
    a = torch.eye(n)
    for i, j in edges:
        a[i, j] = a[j, i] = 1.0
    return a


class SemGraphConv(nn.Module):
    """Semantic graph convolution (learnable edge weights over a fixed adjacency)."""

    def __init__(self, cin, cout, adj):
        super().__init__()
        self.register_buffer("mask", adj > 0)
        self.w_self = nn.Linear(cin, cout)
        self.w_nb = nn.Linear(cin, cout, bias=False)
        self.e = nn.Parameter(torch.ones(int(self.mask.sum())))

    def forward(self, x):  # x: [B,N,C]
        logits = torch.full(self.mask.shape, -9e15, device=x.device, dtype=x.dtype)
        logits[self.mask] = self.e.to(x.dtype)
        a = torch.softmax(logits, dim=1)
        eye = torch.eye(a.shape[0], device=x.device, dtype=x.dtype)
        return self.w_self(x) * a.diagonal()[None, :, None] + (a * (1 - eye)) @ self.w_nb(x)


class GCNBlock(nn.Module):
    def __init__(self, dim, adj):
        super().__init__()
        self.g1, self.g2 = SemGraphConv(dim, dim, adj), SemGraphConv(dim, dim, adj)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x):
        y = F.gelu(self.n1(self.g1(x)))
        return x + F.gelu(self.n2(self.g2(y)))


class PoseInjection(nn.Module):
    """Garment nodes attend to pose nodes (side stream -> main stream)."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, g, p, p_mask=None):
        out, _ = self.attn(self.norm(g), p, p, key_padding_mask=p_mask)
        return g + torch.tanh(self.gate) * out


class KeypointGCN(nn.Module):
    def __init__(self, n_garment=32, n_pose=10, dim=128, levels=4, sym_pairs=None):
        super().__init__()
        ag = adjacency(n_garment, garment_edges(n_garment, sym_pairs))
        ap = adjacency(n_pose, POSE_EDGES)
        self.g_in = nn.Linear(2, dim)
        self.p_in = nn.Linear(5, dim)  # x, y, conf, visibility, depth
        self.g_pos = nn.Parameter(torch.randn(1, n_garment, dim) * 0.02)
        self.p_pos = nn.Parameter(torch.randn(1, n_pose, dim) * 0.02)
        self.g_blocks = nn.ModuleList([GCNBlock(dim, ag) for _ in range(levels)])
        self.p_blocks = nn.ModuleList([GCNBlock(dim, ap) for _ in range(levels)])
        self.inject = nn.ModuleList([PoseInjection(dim) for _ in range(levels)])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def forward(self, kc, pose, vis, depth):
        """kc: [B,Ng,2] in-shop garment keypoints, normalised to [-1,1]
        pose: [B,Np,3] (x,y normalised to [-1,1], conf); vis, depth: [B,Np]
        returns K̂_c: [B,Ng,2] normalised target-pose garment keypoints."""
        p_feat = torch.cat([pose, vis[..., None], depth[..., None]], -1)
        p_missing = pose[..., 2] <= 0
        p_missing[:, 1] = False  # never mask every key (neck kept)
        g = self.g_in(kc) + self.g_pos
        p = self.p_in(p_feat) + self.p_pos
        for gb, pb, inj in zip(self.g_blocks, self.p_blocks, self.inject):
            p = pb(p)
            g = inj(gb(g), p, p_missing)
        return kc + self.head(g)  # residual on the in-shop layout


def normalise_xy(xy, h, w):
    """pixel (x,y) -> [-1,1]; missing points (negative) stay at -2."""
    out = torch.stack([xy[..., 0] / (w - 1) * 2 - 1, xy[..., 1] / (h - 1) * 2 - 1], -1)
    return torch.where(xy < 0, torch.full_like(out, -2.0), out)


def denormalise_xy(xy, h, w):
    return torch.stack([(xy[..., 0] + 1) / 2 * (w - 1), (xy[..., 1] + 1) / 2 * (h - 1)], -1)


def keypoint_loss(pred, gt):
    valid = (gt > -1.5).all(-1, keepdim=True).float()
    return (valid * (pred - gt).abs()).sum() / valid.sum().clamp(min=1) / 2
