"""Pose frames and rotations.

A probe pose is 9 numbers [apex / 100 mm, beam, lateral]; its frame has columns (lateral, beam, normal).
The liver placement p_L = (R_L, t) maps the liver (CT RAS) frame into the frame of the first observed view, P_ref:
R_L rotates liver-frame vectors into P_ref, t is the liver centroid in P_ref (/100 mm).
"""
import numpy as np
import torch
import torch.nn.functional as F

from sonobelief import fan as G


def frame(pose):
    """Liver-frame pose (..., 9) -> apex (..., 3), R (..., 3, 3) columns lateral, beam, normal."""
    a, b, l = pose[..., :3], pose[..., 3:6], pose[..., 6:9]
    return a, torch.stack([l, b, torch.cross(l, b, dim=-1)], -1)


def to_liver(R_L, t, rel):
    """rel: poses in the P_ref frame (..., K, 9) -> liver frame, given R_L (..., 3, 3), t (..., 3)."""
    Rt = R_L.transpose(-1, -2)[..., None, :, :]
    f = lambda v: (Rt @ v[..., None])[..., 0]
    return torch.cat([f(rel[..., :3] - t[..., None, :]), f(rel[..., 3:6]), f(rel[..., 6:9])], -1)


def placement(pose_ref):
    """True p_L for reference poses (liver frame): R_L = R0^T, t = -R0^T a0."""
    a0, R0 = frame(pose_ref)
    R_L = R0.transpose(-1, -2)
    return R_L, -(R_L @ a0[..., None])[..., 0]


def relative(pose, pose_ref):
    """Liver-frame poses (..., K, 9) -> P_ref frame, with the reference pose (..., 9)."""
    a0, R0 = frame(pose_ref)
    Rt = R0.transpose(-1, -2)[..., None, :, :]
    f = lambda v: (Rt @ v[..., None])[..., 0]
    return torch.cat([f(pose[..., :3] - a0[..., None, :]), f(pose[..., 3:6]), f(pose[..., 6:9])], -1)


def rot_err_deg(A, B):
    c = ((A @ B.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.arccos(c.clamp(-1, 1)))


def gram_schmidt(c0, c1):
    a = F.normalize(c0, dim=-1)
    b = F.normalize(c1 - (a * c1).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


def proj_mean(R):
    """Projected (chordal) mean of rotations (..., M, 3, 3)."""
    U, _, Vh = torch.linalg.svd(R.mean(-3))
    d = torch.det(U @ Vh)
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return U @ D @ Vh


def plane_rot_dist(Ri, R):
    """Rotation distance (deg) between probe frames, invariant to the 180 deg flip about the beam (roll is sampled
    in [0, 180), and roll + 180 shows the same image plane mirrored)."""
    F = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=R.device))
    E = Ri.expand(len(R), -1, -1)
    return torch.minimum(rot_err_deg(E, R), rot_err_deg(E, R @ F))


def canonical(beam, lateral):
    """-> lateral in the theta in [0, pi) convention, and a bool 'flipped' per pose."""
    e2 = G.lateral_axis(beam, torch.full((len(beam),), np.pi / 2, device=beam.device))
    flip = (lateral * e2).sum(-1) < 0
    return torch.where(flip[:, None], -lateral, lateral), flip
