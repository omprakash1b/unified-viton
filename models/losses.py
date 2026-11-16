"""Shared losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGLoss(nn.Module):
    """Perceptual loss on VGG19 relu1_2..relu5_2 (inputs in [-1,1])."""

    def __init__(self, weights=(1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0)):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19
        feats = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().requires_grad_(False)
        cuts = [4, 9, 18, 27, 36]
        self.slices = nn.ModuleList([feats[a:b] for a, b in zip([0] + cuts[:-1], cuts)])
        self.w = weights
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = ((x + 1) / 2 - self.mean) / self.std
        y = ((y + 1) / 2 - self.mean) / self.std
        loss = 0.0
        for w, s in zip(self.w, self.slices):
            x, y = s(x), s(y)
            loss = loss + w * F.l1_loss(x, y.detach())
        return loss


def masked_l1(x, y, m):
    return ((x - y).abs() * m).sum() / (m.sum() * x.shape[1]).clamp(min=1)
