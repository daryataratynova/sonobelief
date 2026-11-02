"""Ridge and small-MLP regressors used for the z_L -> GB-offset conditional and for probes.
"""
import torch
from torch import nn


def ridge(X, Y, lam=1e-3, intercept_last=True):
    """Ridge solution; with intercept_last the last column (a ones column) is not penalized."""
    X, Y = X.double(), Y.double()
    pen = torch.ones(X.shape[1], device=X.device, dtype=X.dtype)
    if intercept_last:
        pen[-1] = 0
    W = torch.linalg.solve(X.T @ X + lam * len(X) * torch.diag(pen), X.T @ Y)
    return W.float()


def fit_mlp(Xtr, Ytr, Xva, steps=4000, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, Ytr.shape[1])).cuda()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xn, Yn = (Xtr - mu) / sd, (Ytr - ym) / ys
    for _ in range(steps):
        i = torch.randint(len(Xn), (512,), device="cuda")
        loss = ((net(Xn[i]) - Yn[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return net((Xva - mu) / sd) * ys + ym


def ridge_pred(Xtr, ytr, Xte, lam):
    """Standardized X, centred y: the intercept is not penalized."""
    mu, sd, ym = Xtr.mean(0), Xtr.std(0) + 1e-6, ytr.mean(0)
    return ((Xte - mu) / sd) @ ridge((Xtr - mu) / sd, ytr - ym, lam, intercept_last=False) + ym
