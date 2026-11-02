"""Per-split views with frozen encodings: standardized z_L, E_img embeddings and liver-frame poses of the random views,
plus draw / build of random multi-view conditions (the placement flow's pre-training data).
"""
import os

import numpy as np
import torch

from sonobelief.flow import KMAX
from sonobelief.geometry import placement, relative
from sonobelief.paths import ANATOMY, EMB_RUN
from sonobelief.table import load_table

N_PLANES = 512


class Views:
    """Per split: standardized z_L (S, 32), e (S, 512, 128), pose in liver frame (S, 512, 9)."""

    def __init__(self, split, zl_run, stats=None):
        self.studies = list(load_table(split).index)
        z = np.load(os.path.join(zl_run, f"z_L_{split}.npz"))
        assert list(z["study"]) == self.studies
        zl = torch.from_numpy(z["z_L"]).cuda().float()
        emb = np.load(os.path.join(EMB_RUN, f"embeddings_{split}.npz"))
        S = len(self.studies)
        assert (emb["study"].reshape(S, N_PLANES) == np.array(self.studies)[:, None]).all()
        e = torch.from_numpy(emb["z"].astype(np.float32)).cuda()
        if stats is None:
            stats = {"zl": (zl.mean(0), zl.std(0)), "e": (e.mean(0), e.std(0))}
        self.stats = stats
        self.zl = (zl - stats["zl"][0]) / stats["zl"][1]
        self.e = ((e - stats["e"][0]) / stats["e"][1]).view(S, N_PLANES, -1)
        lc = torch.from_numpy(np.stack([np.load(os.path.join(ANATOMY, f"{s}.npz"))["liver_c"]
                                        for s in self.studies])).cuda()
        ap, be, la = (torch.from_numpy(emb[k]).float().cuda().view(S, N_PLANES, 3) for k in ("apex", "beam", "lateral"))
        self.pose = torch.cat([(ap - lc[:, None]) / 100, be, la], -1)
        self.window = torch.from_numpy(emb["window"].reshape(S, N_PLANES).astype(np.int64)).cuda()
        self.r_gb = torch.from_numpy(z["r_gb"]).cuda().float()

    def __len__(self):
        return len(self.studies)


def dice_of(dec, e_std, stats, masks, chunk=2048):
    out = []
    for i in range(0, len(e_std), chunk):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dec(e_std[i:i + chunk] * stats[1] + stats[0])[:, 0].float() > 0
        m = masks[i:i + chunk]
        inter = (pred & m).sum((1, 2)).float()
        s = (pred.sum((1, 2)) + m.sum((1, 2))).float()
        out.append(2 * inter / s.clamp(min=1))
    return torch.cat(out)


def draw(V, pat, gen):
    """pat (B,) -> KMAX distinct liver-aimed view indices per row; column 0 is P_ref."""
    return torch.multinomial((V.window[pat] < 2).float(), KMAX, replacement=False, generator=gen)


def build(V, pat, obs, k):
    """-> tokens (B, KMAX, 137), valid (B, KMAX), raw state w (B, 44)."""
    pose_ref = V.pose[pat, obs[:, 0]]
    rel = relative(V.pose[pat[:, None], obs], pose_ref)
    tok = torch.cat([V.e[pat[:, None], obs], rel], -1)
    valid = torch.arange(KMAX, device="cuda")[None] < k[:, None]
    R_L, t = placement(pose_ref)
    w = torch.cat([t, R_L[..., 0], R_L[..., 1], V.zl[pat], V.r_gb[pat] / 100], -1)
    return tok, valid, w
