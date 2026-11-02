"""Placement flow p(p_L | O_1:K): a set encoder over the observed views and a conditional flow-matching velocity field.

Each view is a token [standardized E_img embedding (128), pose relative to P_ref (9)]. The set encoder adds a learned
null token, so K = 0 (no views) gives the population prior. The state is [t, first two columns of R_L] (9-d),
standardized with train statistics; samples are drawn with a midpoint ODE from tau = 0 to 1.
"""
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from sonobelief import paths

K_SET = (0, 1, 2, 4, 8, 16)
KMAX = max(K_SET)
TOK_DIM = 128 + 9


class SetEncoder(nn.Module):
    def __init__(self, d, layers, heads, n_seeds, dropout):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(TOK_DIM, d), nn.SiLU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.sab = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d) * 0.02)
        self.pma = nn.MultiheadAttention(d, heads, batch_first=True)
        self.kemb = nn.Embedding(KMAX + 1, d)
        self.out = nn.Linear(n_seeds * d, d)

    def forward(self, tok, valid):
        B = len(tok)
        h = torch.cat([self.null.expand(B, -1, -1), self.inp(tok)], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tok.device), ~valid], 1)
        h = self.norm(self.sab(h, src_key_padding_mask=pad))
        c, _ = self.pma(self.seeds.expand(B, -1, -1), h, h, key_padding_mask=pad, need_weights=False)
        return self.out(c.flatten(1)) + self.kemb(valid.sum(1))


class AdaBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.film = nn.Linear(width, 3 * width)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * width, width))

    def forward(self, h, c):
        s, b, g = self.film(c).chunk(3, -1)
        return h + g * self.mlp(self.norm(h) * (1 + s) + b)


class Velocity(nn.Module):
    def __init__(self, w_dim, d, width, depth, dropout):
        super().__init__()
        self.inp = nn.Linear(w_dim, width)
        self.temb = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.cond = nn.Linear(d, width)
        self.blocks = nn.ModuleList([AdaBlock(width, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, w_dim))

    def forward(self, x, tau, C):
        f = tau[:, None] * 1000 * torch.exp(-math.log(1e4) * torch.arange(32, device=x.device) / 32)
        c = F.silu(self.temb(torch.cat([f.sin(), f.cos()], 1)) + self.cond(C))
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, c)
        return self.out(h)


class Posterior(nn.Module):
    def __init__(self, cfg, w_dim):
        super().__init__()
        self.enc = SetEncoder(cfg["d"], cfg["layers"], cfg["heads"], cfg["seeds"], cfg["dropout"])
        self.vel = Velocity(w_dim, cfg["d"], cfg["width"], cfg["depth"], cfg["dropout"])
        self.register_buffer("w_mu", torch.zeros(w_dim))
        self.register_buffer("w_sd", torch.ones(w_dim))

    def loss(self, tok, valid, x1, x0, tau):
        """-> per-example, per-dimension squared error (B, 44)."""
        xt = (1 - tau[:, None]) * x0 + tau[:, None] * x1
        return (self.vel(xt, tau, self.enc(tok, valid)) - (x1 - x0)) ** 2

    @torch.no_grad()
    def sample(self, tok, valid, M, steps, gen):
        """-> raw w (B, M, 44); midpoint ODE from tau = 0 to 1."""
        B = len(tok)
        C = self.enc(tok, valid).repeat_interleave(M, 0)
        x = torch.randn(B * M, len(self.w_mu), device="cuda", generator=gen)
        h = 1 / steps
        for i in range(steps):
            tau = torch.full((B * M,), i * h, device="cuda")
            xm = x + 0.5 * h * self.vel(x, tau, C)
            x = x + h * self.vel(xm, tau + 0.5 * h, C)
        return (x * self.w_sd + self.w_mu).view(B, M, -1)


def load_flow(run):
    ck = torch.load(os.path.join(paths.REPO, run, "best.pt"), weights_only=False)
    m = Posterior(ck["config"], len(ck["model"]["w_mu"])).cuda().eval()
    m.load_state_dict(ck["model"])
    return m, ck
