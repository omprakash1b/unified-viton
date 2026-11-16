"""Stage 2 — target segmentation Ŝ and occlusion map O.

(Ŝ, O) = G_seg(S_a, R(K̂_c), D_p, M_c)
L_seg  = CE(Ŝ, S_gt) + λ_d Dice(Ŝ, S_gt) + λ_o BCE(O, O_gt)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import UNet


class SegNet(nn.Module):
    def __init__(self, n_seg=8, n_garment_kpts=32, n_pose=10, base=64):
        super().__init__()
        cin = n_seg + n_garment_kpts + n_pose + 3 + 1  # S_a, R(K̂_c), pose maps, DensePose, M_c
        self.n_seg = n_seg
        self.net = UNet(cin, n_seg + 1, base=base)

    def forward(self, seg_agnostic, kpt_maps, pose_maps, densepose, cloth_mask):
        x = torch.cat([seg_agnostic, kpt_maps, pose_maps, densepose, cloth_mask], 1)
        out = self.net(x)
        return out[:, : self.n_seg], out[:, self.n_seg:]  # seg logits, occlusion logit


def dice_loss(logits, target, n_cls, eps=1.0):
    prob = logits.softmax(1)
    oh = F.one_hot(target, n_cls).permute(0, 3, 1, 2).float()
    inter = (prob * oh).sum((2, 3))
    union = prob.sum((2, 3)) + oh.sum((2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def seg_loss(seg_logits, occ_logit, seg_gt, occ_gt, n_cls, lam_dice=1.0, lam_occ=1.0):
    ce = F.cross_entropy(seg_logits, seg_gt)
    dl = dice_loss(seg_logits, seg_gt, n_cls)
    pos = occ_gt.mean().clamp(min=1e-3)
    occ = F.binary_cross_entropy_with_logits(occ_logit, occ_gt, pos_weight=(1 - pos) / pos)
    return ce + lam_dice * dl + lam_occ * occ, {"ce": ce.item(), "dice": dl.item(), "occ": occ.item()}
