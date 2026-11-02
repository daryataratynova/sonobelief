"""Sample ultrasound-fan observation planes from every study's liver mask.

For each study, N_PLANES probe poses are drawn and each gives a 2 x IMG x IMG observation:
  channel 0  liver   = liver mask  AND inside the fan
  channel 1  fov     = body        AND inside the fan  (valid ultrasound region)
so "outside the fan" (fov = 0) and "in the fan but no liver" (fov = 1, liver = 0) differ.

Pose types, with fixed quotas per study (MIX; studies without a xiphoid tip give the
subxiphoid share to general):
  window 0  general     target = liver point; the probe sits where the line from the target
                        along a random outward direction leaves the body. Directions span
                        anterior-left (-30 deg) to right posterolateral (110 deg) in the axial
                        plane and -45..+15 deg elevation: subcostal, intercostal, flank windows.
  window 1  subxiphoid  target = liver point; the probe sits on the skin 0-50 mm below and
                        +-30 mm beside the xiphoid tip.
  window 2  off-target  target = random body point within OFF_MARGIN_MM of the liver's
                        bounding box (not necessarily liver); probe placed as in general. Gives
                        frames with little or no liver: uninformative observations.
Liver targets are stratified: the liver bounding box is split into 3x3x3 regions (L/R, P/A,
I/S); a region is chosen uniformly among those holding >= MIN_BIN_FRAC of the liver, then a
voxel inside it. This keeps the large right lobe from dominating.
Then: random rotation about the beam axis (theta in [0, pi)); poses whose target lies deeper
than DEPTH_MM - 10 mm, or whose beam doesn't enter the body, are redrawn.
No acoustic shadowing (ribs, lung, gas): "visible" means geometrically in the plane.

Output: work/planes/<study>.npz
  masks      uint8 (N, 2, IMG, IMG/8), np.packbits along the last axis
  apex, beam, lateral, skin, target  float32 (N, 3)  canonical-RAS mm (see fan.py)
  theta float32 (N,), window uint8 (N,) 0 = general, 1 = subxiphoid, 2 = off-target
plus work/planes/geometry.json and work/qc/planes/<study>.png previews.

Usage: python -m sonobelief.simulate.planes [--n 512] [--studies s0015 ...] [--overwrite]
"""
import argparse
import hashlib
import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from sonobelief import fan as G
from sonobelief import paths
from sonobelief.table import load_table

matplotlib.use("Agg")


OUT = paths.PLANES
PREV = os.path.join(paths.QC, "planes")
SEG_BODY = paths.SEG_BODY
MIX = {"general": 0.55, "subxiphoid": 0.30, "off_target": 0.15}
WINDOW_CODE = {"general": 0, "subxiphoid": 1, "off_target": 2}
NAMES = {v: k for k, v in WINDOW_CODE.items()}
BINS = 3  # liver bounding box split into BINS^3 regions for stratified targets
MIN_BIN_FRAC = 0.01
OFF_MARGIN_MM = 60
AZIMUTH_DEG = (-30, 110)  # 0 = anterior, 90 = patient right
ELEVATION_DEG = (-45, 15)  # negative = probe below the target, beam angled up
MARCH_MM = 400


class Volume:
    """Label volume on the GPU: 0 outside body, 1 body, 2 liver. Sampled at mm coordinates."""

    def __init__(self, row, device):
        liver_img = nib.as_closest_canonical(nib.load(os.path.join(row["seg_dir"], "liver.nii.gz")))
        body = np.asanyarray(nib.as_closest_canonical(nib.load(os.path.join(SEG_BODY, f"{row.name}.nii.gz"))).dataobj) > 0
        liver = np.asanyarray(liver_img.dataobj) > 0
        lab = body.astype(np.uint8)
        lab[liver] = 2
        self.affine = liver_img.affine
        self.sp = torch.tensor(liver_img.header.get_zooms()[:3], device=device, dtype=torch.float32)
        self.shape = torch.tensor(lab.shape, device=device, dtype=torch.float32)
        self.vol = torch.from_numpy(lab).to(device).float()[None, None]  # (1, 1, X, Y, Z)
        self.liver_mm = torch.from_numpy(np.argwhere(liver)).to(device).float() * self.sp

        # stratified liver targets: BINS^3 regions of the liver bounding box
        lo, hi = self.liver_mm.min(0).values, self.liver_mm.max(0).values
        b = ((self.liver_mm - lo) / (hi - lo + 1e-6) * BINS).long().clamp(max=BINS - 1)
        self.liver_bin = (b[:, 0] * BINS + b[:, 1]) * BINS + b[:, 2]
        counts = torch.bincount(self.liver_bin, minlength=BINS**3)
        self.bins_used = torch.nonzero(counts >= MIN_BIN_FRAC * len(self.liver_mm)).flatten()
        order = torch.argsort(self.liver_bin)
        self.liver_sorted = self.liver_mm[order]
        self.bin_start = torch.cumsum(counts, 0) - counts
        self.bin_count = counts

        # off-target targets: body voxels within OFF_MARGIN_MM of the liver bounding box
        lo_v = ((lo - OFF_MARGIN_MM) / self.sp).floor().long().clamp(min=0).tolist()
        hi_v = ((hi + OFF_MARGIN_MM) / self.sp).ceil().long().tolist()
        sub = body[lo_v[0]:hi_v[0] + 1, lo_v[1]:hi_v[1] + 1, lo_v[2]:hi_v[2] + 1]
        self.off_mm = (torch.from_numpy(np.argwhere(sub)).to(device) + torch.tensor(lo_v, device=device)).float() * self.sp

    def liver_targets(self, n, gen):
        """Stratified: uniform over used bins, then uniform voxel within the bin."""
        dev = self.sp.device
        bins = self.bins_used[torch.randint(len(self.bins_used), (n,), generator=gen, device=dev)]
        k = (torch.rand(n, generator=gen, device=dev) * self.bin_count[bins]).long()
        return self.liver_sorted[self.bin_start[bins] + k]

    def off_targets(self, n, gen):
        return self.off_mm[torch.randint(len(self.off_mm), (n,), generator=gen, device=self.sp.device)]

    def world_to_mm(self, xyz):
        vox = np.linalg.inv(self.affine) @ np.r_[xyz, 1]
        return torch.tensor(vox[:3], device=self.sp.device, dtype=torch.float32) * self.sp

    def sample(self, pts):
        """pts (..., 3) mm -> labels (...) (nearest; outside the volume = 0)."""
        shp = pts.shape[:-1]
        n = 2 * (pts.reshape(1, -1, 1, 1, 3) / self.sp) / (self.shape - 1) - 1
        g = n.flip(-1)  # grid_sample wants (W, H, D) = (z, y, x)
        out = F.grid_sample(self.vol, g, mode="nearest", padding_mode="zeros", align_corners=True)
        return out.reshape(shp).round().to(torch.uint8)


def skin_along(vol, start, direction):
    """Last in-body point marching from start along direction (N, 3); NaN where never leaving."""
    r = torch.arange(MARCH_MM, device=start.device, dtype=torch.float32)
    pts = start[:, None, :] + r[None, :, None] * direction[:, None, :]
    inside = vol.sample(pts) > 0
    first_out = torch.argmax((~inside).int(), dim=1)
    ok = (~inside).any(1) & (first_out > 0)
    skin = start + (first_out - 1).clamp(min=0)[:, None].float() * direction
    skin[~ok] = float("nan")
    return skin


def draw_poses(vol, n, kind, tip_mm, gen):
    """n candidate poses of one kind ('general', 'subxiphoid', 'off_target') and a validity mask."""
    dev = vol.vol.device
    rnd = lambda *s: torch.rand(*s, generator=gen, device=dev)
    target = vol.off_targets(n, gen) if kind == "off_target" else vol.liver_targets(n, gen)
    target = target + (rnd(n, 3) - 0.5) * vol.sp  # jitter inside the voxel

    if kind == "subxiphoid":
        off = torch.stack([(rnd(n) - 0.5) * 60, torch.full((n,), -25.0, device=dev), -rnd(n) * 50], -1)
        ant = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
        skin = skin_along(vol, tip_mm[None] + off, ant)
    else:  # march outward from the target
        az = torch.deg2rad(AZIMUTH_DEG[0] + rnd(n) * (AZIMUTH_DEG[1] - AZIMUTH_DEG[0]))
        el = torch.deg2rad(ELEVATION_DEG[0] + rnd(n) * (ELEVATION_DEG[1] - ELEVATION_DEG[0]))
        u = torch.stack([torch.sin(az) * torch.cos(el), torch.cos(az) * torch.cos(el), torch.sin(el)], -1)
        skin = skin_along(vol, target, u)

    beam = target - skin
    dist = beam.norm(dim=-1)
    beam = beam / dist[:, None]
    ok = torch.isfinite(dist) & (dist > 5) & (dist < G.DEPTH_MM - 10)
    ok &= vol.sample(skin + 5 * beam) > 0  # beam enters the body
    theta = rnd(n) * math.pi
    lateral = G.lateral_axis(beam, theta)
    apex = skin - G.R_APEX_MM * beam
    window = torch.full((n,), WINDOW_CODE[kind], dtype=torch.uint8, device=dev)
    return dict(apex=apex, beam=beam, lateral=lateral, skin=skin, target=target,
                theta=theta, window=window), ok


def quotas(n, has_tip):
    mix = dict(MIX)
    if not has_tip:
        mix["general"] += mix.pop("subxiphoid")
    q = {k: int(round(v * n)) for k, v in mix.items()}
    q["general"] += n - sum(q.values())  # rounding remainder
    return q


def sample_study(row, n, device, seed):
    vol = Volume(row, device)
    has_tip = bool(row.sternum_present)
    tip = vol.world_to_mm([row.tip_x_mm, row.tip_y_mm, row.tip_z_mm]) if has_tip else None
    gen = torch.Generator(device=device).manual_seed(seed)
    keys = ["apex", "beam", "lateral", "skin", "target", "theta", "window"]
    parts = {k: [] for k in keys}
    for kind, q in quotas(n, has_tip).items():
        got = {k: [] for k in keys}
        have = 0
        for _ in range(50):
            p, ok = draw_poses(vol, 2 * q, kind, tip, gen)
            for k in keys:
                got[k].append(p[k][ok])
            have += int(ok.sum())
            if have >= q:
                break
        assert have >= q, f"{row.name}: only {have}/{q} valid {kind} poses"
        for k in keys:
            parts[k].append(torch.cat(got[k])[:q])
    poses = {k: torch.cat(v) for k, v in parts.items()}
    perm = torch.randperm(n, generator=gen, device=device)  # interleave window types
    poses = {k: v[perm] for k, v in poses.items()}

    X, Y, in_fan = G.pixel_grid(device)
    masks = []
    for i in range(0, n, 128):
        sl = slice(i, i + 128)
        pts = G.plane_points(poses["apex"][sl], poses["beam"][sl], poses["lateral"][sl], X, Y)
        lab = vol.sample(pts)
        masks.append(torch.stack([(lab == 2) & in_fan, (lab > 0) & in_fan], 1))
    masks = torch.cat(masks).cpu().numpy()
    return masks, {k: v.cpu().numpy() for k, v in poses.items()}


def preview(sid, masks, window, path, k=16):
    fig, axs = plt.subplots(2, 8, figsize=(16, 4.4))
    for a, i in zip(axs.ravel(), np.linspace(0, len(masks) - 1, k).astype(int)):
        img = masks[i, 1] * 0.35 + masks[i, 0] * 0.65
        a.imshow(img, cmap="magma", vmin=0, vmax=1)
        a.set_title(f"#{i} {NAMES[window[i]]}", fontsize=8)
        a.set_axis_off()
    fig.suptitle(f"{sid}: liver (bright) inside FOV (dim); apex at top", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="planes per study")
    ap.add_argument("--studies", nargs="*")
    ap.add_argument("--split", help="only this split (e.g. external, which load_table() does not include)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PREV, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(OUT, "geometry.json"), "w") as f:
        json.dump({k: getattr(G, k) for k in ["R_APEX_MM", "SECTOR_DEG", "DEPTH_MM", "IMG", "PX_MM", "Y0_MM"]}
                  | {"n_planes": args.n, "seed": args.seed, "mix": MIX, "window_code": WINDOW_CODE,
                     "liver_bins": BINS, "min_bin_frac": MIN_BIN_FRAC, "off_margin_mm": OFF_MARGIN_MM,
                     "azimuth_deg": AZIMUTH_DEG, "elevation_deg": ELEVATION_DEG,
                     "channels": ["liver", "fov"]}, f, indent=1)

    table = load_table(args.split)
    if args.studies:
        table = table.loc[args.studies]
    for k, (sid, row) in enumerate(table.iterrows()):
        out = os.path.join(OUT, f"{sid}.npz")
        if os.path.exists(out) and not args.overwrite:
            continue
        seed = args.seed * 1_000_003 + int(hashlib.md5(sid.encode()).hexdigest()[:8], 16)
        masks, poses = sample_study(row, args.n, device, seed)
        np.savez_compressed(out, masks=np.packbits(masks, axis=-1), **poses)
        preview(sid, masks, poses["window"], os.path.join(PREV, f"{sid}.png"))
        lf = masks[:, 0].sum((1, 2)) / masks[:, 1].sum((1, 2)).clip(1)
        w = poses["window"]
        print(f"[{k + 1}/{len(table)}] {sid}: liver/FOV median {np.median(lf[w < 2]):.2f} (liver-aimed), "
              f"{np.median(lf[w == 2]):.2f} (off-target, {np.mean(lf[w == 2] == 0):.0%} empty); "
              f"windows {np.bincount(w, minlength=3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
