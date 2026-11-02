"""GB geometry state (centroid offset r_GB, long axis a_GB, length l_GB) and scoring with a geometry-conditioned head.

The head state is [r / 100, a a^T (6 unique entries), l / 100]; a a^T makes it invariant to the sign of a.
"""
import os

import numpy as np
import torch

from sonobelief.regression import ridge_pred
from sonobelief.target_poses import BANKS

LAM_R, LAMS, FOLDS, M = 0.1, (0.1, 1.0, 10.0, 100.0, 1000.0), 10, 32
IU = torch.triu_indices(3, 3)


def geometry(V, idx):
    """True (r mm, Q6, log l) for patients idx -> (n, 10)."""
    out = []
    for i in idx.tolist():
        d = np.load(os.path.join(BANKS["dense"], f"{V.studies[i]}.npz"))
        a = torch.from_numpy(d["gb_axes"][0]).float().cuda()
        out.append(torch.cat([V.r_gb[i], torch.outer(a, a)[IU[0], IU[1]],
                              torch.tensor([np.log(float(d["gb_length"]))], device="cuda").float()]))
    return torch.stack(out)


def oof(X, Y, lam, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(len(X), generator=g).cuda()
    out = torch.zeros_like(Y)
    for f in range(FOLDS):
        te = perm[f::FOLDS]
        tr = perm[~torch.isin(perm, te)]
        out[te] = ridge_pred(X[tr], Y[tr], X[te], lam)
    return out


def axis_of(q6):
    Q = torch.zeros(*q6.shape[:-1], 3, 3, device=q6.device)
    Q[..., IU[0], IU[1]] = q6
    Q = Q + Q.transpose(-1, -2) - torch.diag_embed(Q.diagonal(dim1=-2, dim2=-1))
    return torch.linalg.eigh(Q)[1][..., -1]


def angle(a, b):
    return torch.rad2deg(torch.arccos((a * b).sum(-1).abs().clamp(max=1)))


def state_of(y):
    """(..., 10) [r, Q6, log l] -> head state (..., 10) [r / 100, a a^T (6), l / 100] with a re-normalized."""
    a = axis_of(y[..., 3:9])
    q = (a[..., :, None] * a[..., None, :])[..., IU[0], IU[1]]
    return torch.cat([y[..., :3] / 100, q, torch.exp(y[..., 9:10]) / 100], -1)


def quality(Y, P):
    a_t, a_p = axis_of(Y[:, 3:9]), axis_of(P[:, 3:9])
    return {"r_gb_err_mm": (Y[:, :3] - P[:, :3]).norm(dim=1).mean().item(),
            "axis_err_deg": angle(a_t, a_p).mean().item(), "axis_err_deg_median": angle(a_t, a_p).median().item(),
            "length_err_mm": (torch.exp(Y[:, 9]) - torch.exp(P[:, 9])).abs().mean().item()}


@torch.no_grad()
def scores(head, D, states):
    """states (S, M, 10) per patient -> mean over samples of sigmoid g, per pose of D."""
    head.eval()
    out = torch.zeros(len(D), device="cuda")
    for i in range(0, len(D), 16384):
        sl = slice(i, i + 16384)
        st = states[D.pat[sl]]                                                      # (b, M, 10)
        b, m = st.shape[:2]
        lo = head(st.reshape(b * m, -1), D.pose[sl].repeat_interleave(m, 0)[:, None])[:, 0, 0].view(b, m)
        out[sl] = torch.sigmoid(lo).mean(1)
    return out
