"""Ranking metrics over a candidate bank: AUROC (ties take the average rank) and average precision (AUPRC).
"""
import torch


def auroc(s, y):
    """AUROC of scores s for binary labels y (ties: average rank)."""
    r = torch.empty_like(s)
    o = torch.argsort(s)
    r[o] = torch.arange(1, len(s) + 1, device=s.device, dtype=s.dtype)
    _, inv, cnt = torch.unique(s, return_inverse=True, return_counts=True)   # average ranks over ties
    r = torch.zeros(len(cnt), device=s.device).index_add_(0, inv, r)[inv] / cnt[inv]
    npos = y.sum()
    return ((r[y > 0].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))).item()


def avg_precision(s, y):
    o = torch.argsort(s, descending=True)
    yo = y[o]
    hits = torch.cumsum(yo, 0)
    prec = hits / torch.arange(1, len(yo) + 1, device=s.device)
    return (prec * yo).sum().item() / max(yo.sum().item(), 1)
