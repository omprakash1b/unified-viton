"""Small convolutional U-Net used by Stage 2 (segmentation) and Stage 3 (flow refinement)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
    )


class UNet(nn.Module):
    def __init__(self, cin, cout, base=64, depth=4, return_features=False):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.inc = conv_block(cin, chs[0])
        self.downs = nn.ModuleList([conv_block(chs[i], chs[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([conv_block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth))])
        self.out = nn.Conv2d(chs[0], cout, 1)
        self.return_features = return_features

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(F.avg_pool2d(skips[-1], 2)))
        y = skips.pop()
        for u in self.ups:
            s = skips.pop()
            y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
            y = u(torch.cat([y, s], 1))
        return (self.out(y), y) if self.return_features else self.out(y)
