"""Building blocks and the observation model D_obs(z_L, P^L) -> standardized E_img embedding of the view at pose P.
"""
import math

import torch
from torch import nn


def block3d(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 4, 2, 1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv3d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.SiLU())


def mlp(din, dh, dout, n=2):
    layers, d = [], din
    for _ in range(n):
        layers += [nn.Linear(d, dh), nn.SiLU()]
        d = dh
    return nn.Sequential(*layers, nn.Linear(d, dout))


class Fourier(nn.Module):
    """x (..., d) -> [x, sin(2^k pi x), cos(2^k pi x)] for k < n_freq."""

    def __init__(self, n_freq=6):
        super().__init__()
        self.register_buffer("f", (2.0 ** torch.arange(n_freq)) * math.pi)

    def forward(self, x):
        a = x[..., None] * self.f
        return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)

    def dim(self, d):
        return d * (1 + 2 * len(self.f))


class DObs(nn.Module):
    """Residual MLP on [z, pose features]; FiLM-free, concatenation only."""

    def __init__(self, z_dim=64, e_dim=128, width=512, depth=4, n_freq=6):
        super().__init__()
        self.ff = Fourier(n_freq)
        self.pose = mlp(self.ff.dim(9), width, width, 1)
        self.zin = nn.Linear(z_dim, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.SiLU(),
                                                   nn.Linear(2 * width, width)) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, e_dim))

    def forward(self, z, pose):
        """z (B, z_dim), pose (B, K, 9) -> (B, K, e_dim)."""
        h = self.pose(self.ff(pose)) + self.zin(z)[:, None]
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)
