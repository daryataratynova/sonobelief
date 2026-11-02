"""Ultrasound fan geometry shared by plane sampling and downstream code.

Convex probe, same parameters as us_mask_from_ct.py: virtual apex R_APEX_MM behind the
skin, SECTOR_DEG total opening, DEPTH_MM imaging depth from the skin.

Observation image: IMG x IMG pixels, isotropic PX_MM. Column = lateral position (centred on
the beam axis), row = axial distance from the apex, starting at the skin arc's top
(Y0_MM = R_APEX * cos(half sector)). The fan is wider than it is deep, so the bottom rows
are always outside the fan.

All coordinates are millimetres in the CT's canonical (nibabel RAS+) voxel frame:
voxel index * spacing, +x Right, +y Anterior, +z Superior, origin at voxel (0, 0, 0).
"""
import math

import torch

R_APEX_MM = 50.0
SECTOR_DEG = 65.0
DEPTH_MM = 170.0
IMG = 128

HALF = math.radians(SECTOR_DEG / 2)
R_MAX = R_APEX_MM + DEPTH_MM
WIDTH_MM = 2 * R_MAX * math.sin(HALF)  # fan width at full depth
PX_MM = WIDTH_MM / IMG
Y0_MM = R_APEX_MM * math.cos(HALF)


def pixel_grid(device="cpu"):
    """Lateral X and axial Y (mm, apex-relative) of every pixel centre, plus the fan mask."""
    j = torch.arange(IMG, device=device, dtype=torch.float32)
    X = (j - (IMG - 1) / 2) * PX_MM
    Y = Y0_MM + (j + 0.5) * PX_MM
    Y, X = torch.meshgrid(Y, X, indexing="ij")  # rows = axial, cols = lateral
    r = torch.sqrt(X**2 + Y**2)
    ang = torch.atan2(X, Y)
    in_fan = (r >= R_APEX_MM) & (r <= R_MAX) & (ang.abs() <= HALF)
    return X, Y, in_fan


def plane_points(apex, beam, lateral, X, Y):
    """World points (N, IMG, IMG, 3) for poses given as apex (N,3), beam (N,3), lateral (N,3)."""
    return (apex[:, None, None, :]
            + Y[None, :, :, None] * beam[:, None, None, :]
            + X[None, :, :, None] * lateral[:, None, None, :])


def lateral_axis(beam, theta):
    """Unit vector perpendicular to beam, rotated by theta (rad) about the beam axis.
    theta = 0 puts the lateral axis as close as possible to superior (+z): a longitudinal view."""
    up = torch.tensor([0.0, 0.0, 1.0], device=beam.device).expand_as(beam)
    alt = torch.tensor([1.0, 0.0, 0.0], device=beam.device).expand_as(beam)
    ref = torch.where((beam * up).sum(-1, keepdim=True).abs() > 0.95, alt, up)
    e1 = ref - (ref * beam).sum(-1, keepdim=True) * beam
    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    e2 = torch.cross(beam, e1, dim=-1)
    return torch.cos(theta)[:, None] * e1 + torch.sin(theta)[:, None] * e2
