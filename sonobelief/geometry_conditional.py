"""p(r_GB, a_GB, l_GB | z_L) for the geometry-aware head: a regression per block (r: ridge, Q = a a^T: kernel ridge or
ridge, log l: ridge), with hyperparameters chosen by out-of-fold error on train patients only.
"""
import torch

from sonobelief.gb_geometry import FOLDS, angle, axis_of
from sonobelief.regression import ridge_pred

LAM = {"r": 0.1, "Q": 0.1, "logl": 1.0}
BLOCKS = {"r": slice(0, 3), "Q": slice(3, 9), "logl": slice(9, 10)}
KRR_BW, KRR_LAM = (0.5, 1.0, 2.0, 4.0), (0.01, 0.1, 1.0, 10.0)


def folds(n, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g).cuda()
    return [perm[f::FOLDS] for f in range(FOLDS)]


def krr_fit_pred(Xtr, Ytr, Xte, bw, lam):
    mu, sd, ym = Xtr.mean(0), Xtr.std(0) + 1e-6, Ytr.mean(0)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    h = bw * torch.cdist(A, A).median()
    Kmat = lambda U, W: torch.exp(-torch.cdist(U, W) ** 2 / (2 * h ** 2))
    alpha = torch.linalg.solve(Kmat(A, A) + lam * torch.eye(len(A), device="cuda"), Ytr - ym)
    return Kmat(B, A) @ alpha + ym


def oof(fn, X, Y):
    out = torch.zeros_like(Y)
    for te in folds(len(X)):
        tr = torch.ones(len(X), dtype=torch.bool, device="cuda")
        tr[te] = False
        out[te] = fn(X[tr], Y[tr], X[te])
    return out


def predictors(axis, Xtr, Ytr):
    """-> per-block fit/predict functions (Xtr, Ytr, Xte) -> pred, and the chosen Q hyperparameters."""
    fns = {k: (lambda l: (lambda a, b, c: ridge_pred(a, b, c, l)))(LAM[k]) for k in BLOCKS}
    info = {"Q": {"model": "ridge", "lambda": LAM["Q"]}}
    if axis == "krr":
        Yq, a_t = Ytr[:, BLOCKS["Q"]], axis_of(Ytr[:, BLOCKS["Q"]])
        grid = {}
        for bw in KRR_BW:
            for lam in KRR_LAM:
                P = oof(lambda a, b, c: krr_fit_pred(a, b, c, bw, lam), Xtr, Yq)
                grid[(bw, lam)] = angle(a_t, axis_of(P)).mean().item()
        bw, lam = min(grid, key=grid.get)
        print("KRR train OOF axis error (deg): " + ", ".join(f"bw {b:g} lam {l:g}: {e:.1f}" for (b, l), e in grid.items()))
        fns["Q"] = lambda a, b, c: krr_fit_pred(a, b, c, bw, lam)
        info["Q"] = {"model": "krr", "bandwidth_x_median": bw, "lambda": lam, "grid_oof_axis_deg": {f"{b}_{l}": e for (b, l), e in grid.items()}}
    return fns, info
