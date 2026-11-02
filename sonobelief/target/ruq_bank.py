"""Long-axis gallbladder target neighbourhood T_GB^long per CT patient (ground truth; anatomy is used on purpose).

GB geometry (PCA of the CT GB voxels, mm): centroid c, long axis a1 (largest eigenvalue), length L = 2nd-98th
percentile extent of the voxels along a1, widths along a2 / a3 likewise.

Candidate poses (N_POSES per patient, the probe of fan_geometry: convex, 65 deg, 170 mm depth):
  skin contact  body-surface voxel within SKIN_RADIUS_MM of c and not posterior to c - BACK_MARGIN_MM (no back
                windows); right upper quadrant around the GB
  beam          towards c + N(0, AIM_JITTER_MM^2 I); rejected if > MAX_NORMAL_DEG from the inward skin normal,
                if the target is deeper than the imaging depth, or if the beam doesn't enter the body
  roll          uniform in [0, 180) deg about the beam
Each pose is rendered at PX_MM (body / liver / GB in the fan) and scored:
  area         GB area in the fan (mm^2)
  area_rel     area / the patient's largest GB cut over all candidates
  center_deg   |angle of the GB pixel centroid from the beam axis| (deg)
  clip_frac    fraction of GB pixels within EDGE_MM of the fan border (sector edges or the far arc)
  depth_mm     GB pixel centroid depth below the skin (mm)
  align_deg    angle between a1 and the image plane (0 = long axis in plane)
  phi_deg      presentation angle: in-plane long axis vs beam, folded to [0, 90] (0 = along the beam, 90 = across)
  long_frac    extent of the GB pixels along the in-plane projection of a1 / L
Membership of the target T_GB^LA (four components V/A/C/S, thresholds in configs/target_la.json) is decided
by definition.label from these measurements; see target/definition.py.
Acoustic shadowing: along every fan ray (from the skin arc outwards, 1 mm steps) everything beyond the first
blocker is shadowed. Blockers: bone (TotalSegmentator ribs, sternum, vertebrae, scapulae, claviculae; costal
cartilage transmits and is not a blocker) and air (lung lobes, trachea, and CT voxels < GAS_HU inside the body,
which catches bowel gas). Measurements are stored on the visible (unshadowed) GB pixels and on all GB pixels
(*_geo); shadow_frac = shadowed share of the GB pixels in the fan.

Output: work/gb_target/<study>.npz (poses, measurements, component flags, in_T, GB PCA),
        work/gb_target_summary.csv, work/qc/gb_target/<study>.png + index.html (train/val only)
Usage: python -m sonobelief.target.ruq_bank [--split train] [--n 20] [--studies s0004 ...] [--all] [--no_figures]
       python -m sonobelief.target.ruq_bank --figures_only --all --split train     # redraw from stored data
"""
import argparse
import html
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sonobelief import fan as G
from sonobelief import paths
from sonobelief.table import load_table
from sonobelief.target.definition import COMPONENTS, label, load_config
from sonobelief.target.locked import locked

matplotlib.use("Agg")


OUT = os.path.join(paths.WORK, "gb_target")
QC = os.path.join(paths.QC, "gb_target")
SEG_BODY = paths.SEG_BODY

# candidate poses
N_POSES, SKIN_RADIUS_MM, BACK_MARGIN_MM, AIM_JITTER_MM, MAX_NORMAL_DEG = 4000, 150.0, 50.0, 15.0, 60.0
# rendering
PX_MM = 1.0
EDGE_MM = 2.0   # band along the fan border that counts as "clipped" (measurement definition)

LAB_BODY, LAB_LIVER, LAB_GB = 1, 2, 3
# acoustic blockers
BONE_PREFIXES = ("rib_", "sternum", "vertebrae_", "scapula_", "clavicula_")
AIR_MASKS = ("lung_", "trachea")
GAS_HU = -400.0
BLK_BONE, BLK_AIR = 1, 2
N_RAYS = 200


def fan_grid(px=PX_MM):
    """Lateral X, axial Y (apex-relative mm) of a px-mm grid covering the fan; in-fan mask; edge band."""
    xs = torch.arange(-G.R_MAX * math.sin(G.HALF), G.R_MAX * math.sin(G.HALF) + px, px, device="cuda")
    ys = torch.arange(G.Y0_MM, G.R_MAX + px, px, device="cuda")
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    r, ang = torch.sqrt(X ** 2 + Y ** 2), torch.atan2(X, Y)
    in_fan = (r >= G.R_APEX_MM) & (r <= G.R_MAX) & (ang.abs() <= G.HALF)
    d_edge = torch.minimum(G.R_MAX - r, r * torch.sin(G.HALF - ang.abs()))   # to the far arc / to a sector edge
    # polar lookup for shadows: ray index (N_RAYS over the sector) and 1 mm radial step from the skin arc
    ia = ((ang.clamp(-G.HALF, G.HALF) + G.HALF) / (2 * G.HALF) * (N_RAYS - 1)).round().long()
    ir = (r - G.R_APEX_MM).round().long().clamp(0, int(G.DEPTH_MM))
    return X, Y, in_fan, in_fan & (d_edge <= EDGE_MM), (ia, ir)


class Patient:
    def __init__(self, row):
        seg = lambda n: nib.as_closest_canonical(nib.load(os.path.join(row["seg_dir"], n)))
        gb_img = seg("gallbladder.nii.gz")
        gb = np.asanyarray(gb_img.dataobj) > 0
        liver = np.asanyarray(seg("liver.nii.gz").dataobj) > 0
        body = np.asanyarray(nib.as_closest_canonical(nib.load(os.path.join(SEG_BODY, f"{row.name}.nii.gz"))).dataobj) > 0
        lab = body.astype(np.uint8) * LAB_BODY
        lab[liver] = LAB_LIVER
        lab[gb] = LAB_GB
        self.sp = torch.tensor(gb_img.header.get_zooms()[:3], device="cuda", dtype=torch.float32)
        self.shape = torch.tensor(lab.shape, device="cuda", dtype=torch.float32)
        self.lab = torch.from_numpy(lab).cuda().float()[None, None]
        self.lab_np, self.sp_np = lab, gb_img.header.get_zooms()[:3]
        names = os.listdir(row["seg_dir"])
        load = lambda n: np.asanyarray(seg(n).dataobj) > 0
        with ThreadPoolExecutor(8) as ex:
            bone = list(ex.map(load, [n for n in names if n.startswith(BONE_PREFIXES)]))
            air = list(ex.map(load, [n for n in names if n.startswith(AIR_MASKS)]))
        ct = np.asanyarray(nib.as_closest_canonical(nib.load(row["ct"])).dataobj)
        blk = np.zeros(lab.shape, np.uint8)
        blk[np.logical_or.reduce(air) | ((ct < GAS_HU) & body)] = BLK_AIR
        blk[np.logical_or.reduce(bone)] = BLK_BONE
        self.blk = torch.from_numpy(blk).cuda().float()[None, None]
        self.gas_ml = float(((ct < GAS_HU) & body & ~np.logical_or.reduce(air)).sum() * np.prod(gb_img.header.get_zooms()[:3]) / 1000)
        bodyf = torch.from_numpy(body).cuda().float()[None, None]
        self.body = bodyf
        self.smooth = F.avg_pool3d(bodyf, 7, 1, 3)
        er = -F.max_pool3d(-bodyf, 3, 1, 1)
        self.surface = torch.nonzero((bodyf - er)[0, 0] > 0).float() * self.sp
        # GB PCA
        v = torch.from_numpy(np.argwhere(gb)).cuda().float() * self.sp
        self.c = v.mean(0)
        ev, evec = torch.linalg.eigh(torch.cov((v - self.c).T))
        self.axes = evec[:, [2, 1, 0]].T                       # rows a1 (long), a2, a3
        proj = (v - self.c) @ self.axes.T
        q = torch.quantile(proj, torch.tensor([0.02, 0.98], device="cuda"), dim=0)
        self.length, self.widths = (q[1, 0] - q[0, 0]).item(), (q[1, 1:] - q[0, 1:]).tolist()
        self.gb_ml = gb.sum() * float(np.prod(self.sp_np)) / 1000

    def sample(self, vol, pts, mode="nearest"):
        shp = pts.shape[:-1]
        n = 2 * (pts.reshape(1, -1, 1, 1, 3) / self.sp) / (self.shape - 1) - 1
        return F.grid_sample(vol, n.flip(-1), mode=mode, padding_mode="zeros", align_corners=True).reshape(shp)

    def candidates(self, n, gen):
        rnd = lambda *s: torch.rand(*s, device="cuda", generator=gen)
        d = (self.surface - self.c).norm(dim=1)
        cand = self.surface[(d <= SKIN_RADIUS_MM) & (self.surface[:, 1] >= self.c[1] - BACK_MARGIN_MM)]
        out, have = {k: [] for k in ("apex", "beam", "lateral", "skin", "target")}, 0
        for _ in range(100):
            m = 4 * n
            skin = cand[torch.randint(len(cand), (m,), device="cuda", generator=gen)] + (rnd(m, 3) - 0.5) * self.sp
            target = self.c + AIM_JITTER_MM * torch.randn(m, 3, device="cuda", generator=gen)
            beam = target - skin
            dist = beam.norm(dim=1)
            beam = beam / dist[:, None]
            grad = torch.stack([self.sample(self.smooth, skin + 3 * e, "bilinear") - self.sample(self.smooth, skin - 3 * e, "bilinear")
                                for e in torch.eye(3, device="cuda")], -1)
            inward = F.normalize(grad, dim=-1)
            ok = (grad.norm(dim=-1) > 1e-3) & ((beam * inward).sum(-1) >= math.cos(math.radians(MAX_NORMAL_DEG)))
            ok &= (dist < G.DEPTH_MM - 10) & (self.sample(self.body, skin + 5 * beam) > 0.5)
            lateral = G.lateral_axis(beam, rnd(m) * math.pi)
            for k, v in (("apex", skin - G.R_APEX_MM * beam), ("beam", beam), ("lateral", lateral), ("skin", skin), ("target", target)):
                out[k].append(v[ok])
            have += int(ok.sum())
            if have >= n:
                break
        return {k: torch.cat(v)[:n] for k, v in out.items()}

    def render(self, P, X, Y, vol=None):
        """Label images (B, H, W) for poses P."""
        pts = P["apex"][:, None, None] + Y[None, ..., None] * P["beam"][:, None, None] + X[None, ..., None] * P["lateral"][:, None, None]
        return self.sample(self.lab if vol is None else vol, pts).round().to(torch.uint8)

    def shadow(self, P, polar):
        """(B, H, W) bool: behind the first bone / air voxel along the pixel's ray (the blocker itself included)."""
        th = torch.linspace(-G.HALF, G.HALF, N_RAYS, device="cuda")
        r = G.R_APEX_MM + torch.arange(int(G.DEPTH_MM) + 1, device="cuda", dtype=torch.float32)
        d = torch.sin(th)[:, None, None] * P["lateral"][None] + torch.cos(th)[:, None, None] * P["beam"][None]  # (rays, B, 3)
        pts = P["apex"][None, :, None] + r[None, None, :, None] * d[:, :, None]                                     # (rays, B, R, 3)
        hit = (self.sample(self.blk, pts) > 0.5).permute(1, 0, 2)                                                   # (B, rays, R)
        sh = torch.cummax(hit.int(), dim=2).values.bool()
        ia, ir = polar
        return sh[:, ia, ir]


def image_metrics(m, P, a1, length, X, Y, edge):
    """GB pixel mask m (b, H, W) -> area, center_deg, clip_frac, depth_mm, long_frac (b,)."""
    cnt = m.sum((1, 2)).float()
    w = m.float() / cnt.clamp(min=1)[:, None, None]
    xc, yc = (w * X).sum((1, 2)), (w * Y).sum((1, 2))
    normal = torch.cross(P["lateral"], P["beam"], dim=-1)
    a_in = a1 - (a1 * normal).sum(-1, keepdim=True) * normal                      # a1 projected into each plane
    a_x, a_y = (a_in * P["lateral"]).sum(-1), (a_in * P["beam"]).sum(-1)
    nrm = torch.sqrt(a_x ** 2 + a_y ** 2).clamp(min=1e-6)
    t = X[None] * (a_x / nrm)[:, None, None] + Y[None] * (a_y / nrm)[:, None, None]
    big = torch.tensor(1e9, device="cuda")
    ext = torch.where(m, t, -big).amax((1, 2)) - torch.where(m, t, big).amin((1, 2))
    return {"area": cnt * PX_MM ** 2, "center_deg": torch.rad2deg(torch.atan2(xc, yc).abs()),
            "clip_frac": (m & edge).sum((1, 2)).float() / cnt.clamp(min=1),
            "depth_mm": torch.sqrt(xc ** 2 + yc ** 2) - G.R_APEX_MM,
            "long_frac": torch.where(cnt > 0, ext, torch.zeros_like(ext)) / length}


def score(pt, P, X, Y, in_fan, edge, polar, chunk=128):
    """Per-pose criteria values on the visible GB pixels; the same on all GB pixels as *_geo; shadow_frac."""
    out = {}
    a1 = pt.axes[0]
    for i in range(0, len(P["apex"]), chunk):
        Pc = {k: v[i:i + chunk] for k, v in P.items()}
        geo = (pt.render(Pc, X, Y) == LAB_GB) & in_fan                          # (b, H, W)
        vis = geo & ~pt.shadow(Pc, polar)
        r = image_metrics(vis, Pc, a1, pt.length, X, Y, edge)
        r.update({f"{k}_geo": v for k, v in image_metrics(geo, Pc, a1, pt.length, X, Y, edge).items()})
        r["shadow_frac"] = 1 - r["area"] / r["area_geo"].clamp(min=1e-6)
        r["align_deg"] = torch.rad2deg(torch.asin((a1 * torch.cross(Pc["lateral"], Pc["beam"], dim=-1)).sum(-1).abs().clamp(max=1)))
        r["phi_deg"] = torch.rad2deg(torch.atan2((Pc["lateral"] @ a1).abs(), (Pc["beam"] @ a1).abs()))   # presentation angle
        for k, v in r.items():
            out.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- inspection figure

CMAP = np.array([[1, 1, 1], [0.86, 0.86, 0.84], [0.80, 0.62, 0.52], [0.16, 0.62, 0.36]])   # out, body, liver, GB


def slice_panel(ax, pt, plane, P, valid, title):
    """Orthogonal slice through the GB centroid with the long axis and the skin contacts (valid / not)."""
    ci = (pt.c / pt.sp).round().long().tolist()
    sp = pt.sp_np
    lab = pt.lab_np
    if plane == "axial":   # x right, y anterior; display x horizontal, y vertical
        img, dims = lab[:, :, ci[2]], (0, 1)
    elif plane == "coronal":
        img, dims = lab[:, ci[1], :], (0, 2)
    else:
        img, dims = lab[ci[0], :, :], (1, 2)
    ext = [0, img.shape[0] * sp[dims[0]], 0, img.shape[1] * sp[dims[1]]]
    ax.imshow(CMAP[img.T], origin="lower", extent=ext, interpolation="nearest")
    sk = P["skin"].cpu().numpy()
    ax.scatter(sk[~valid, dims[0]], sk[~valid, dims[1]], s=1, c="#b0b0aa", alpha=0.25, lw=0)
    ax.scatter(sk[valid, dims[0]], sk[valid, dims[1]], s=3, c="#2a78d6", alpha=0.6, lw=0)
    c, a1 = pt.c.cpu().numpy(), pt.axes[0].cpu().numpy()
    e = np.stack([c - a1 * pt.length / 2, c + a1 * pt.length / 2])
    ax.plot(e[:, dims[0]], e[:, dims[1]], color="#eb6834", lw=2)
    ax.set_xlim(c[dims[0]] - 170, c[dims[0]] + 170)
    ax.set_ylim(c[dims[1]] - 170, c[dims[1]] + 170)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xticks([]); ax.set_yticks([])


def fan_panel(ax, pt, Pi, X, Y, in_fan, polar, s_i, title):
    img = pt.render(Pi, X, Y)[0].cpu().numpy()
    blk = pt.render(Pi, X, Y, pt.blk)[0].cpu().numpy()
    sh = pt.shadow(Pi, polar)[0].cpu().numpy()
    rgb = CMAP[img].copy()
    rgb[sh] *= 0.45                                   # acoustic shadow
    rgb[blk == BLK_BONE] = (0.98, 0.95, 0.80)         # bone
    rgb[(blk == BLK_AIR) & (img > 0)] = (0.15, 0.15, 0.18)   # gas / lung inside the body
    rgb[~in_fan.cpu().numpy()] = 1.0
    ax.imshow(rgb, extent=[X.min().item(), X.max().item(), Y.max().item(), Y.min().item()], interpolation="nearest")
    t = torch.linspace(-G.HALF, G.HALF, 60, device="cpu")
    for r in (G.R_APEX_MM, G.R_MAX):
        ax.plot(r * torch.sin(t), r * torch.cos(t), color="#8a8a85", lw=0.6)
    for sgn in (-1, 1):
        ax.plot([G.R_APEX_MM * math.sin(sgn * G.HALF), G.R_MAX * math.sin(sgn * G.HALF)],
                [G.R_APEX_MM * math.cos(G.HALF), G.R_MAX * math.cos(G.HALF)], color="#8a8a85", lw=0.6)
    # in-plane projection of the long axis through the GB centroid (projected)
    a1, c = pt.axes[0], pt.c
    nrm = torch.cross(Pi["lateral"][0], Pi["beam"][0], dim=-1)
    v = c - Pi["apex"][0]
    cx, cy = (v * Pi["lateral"][0]).sum().item(), (v * Pi["beam"][0]).sum().item()
    ax_, ay_ = (a1 * Pi["lateral"][0]).sum().item(), (a1 * Pi["beam"][0]).sum().item()
    h = pt.length / 2
    ax.plot([cx - ax_ * h, cx + ax_ * h], [cy - ay_ * h, cy + ay_ * h], color="#eb6834", lw=1.5)
    ax.set_xlim(-G.R_MAX * math.sin(G.HALF) - 5, G.R_MAX * math.sin(G.HALF) + 5)
    ax.set_ylim(G.R_MAX + 5, G.R_APEX_MM * math.cos(G.HALF) - 5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7, loc="left")
    ax.text(0.02, 0.02, f"GB area {s_i['area']:.0f} mm², shadowed {s_i['shadow_frac']:.0%}, depth {s_i['depth_mm']:.0f}\nalign {s_i['align_deg']:.0f}°  "
                        f"long {s_i['long_frac']:.2f}  ctr {s_i['center_deg']:.0f}°  clip {s_i['clip_frac']:.2f}",
            transform=ax.transAxes, fontsize=6, va="bottom")


def figure(pt, sid, P, s, comp, X, Y, in_fan, polar, path, rng, cfg_status=""):
    """s: measurements (numpy); comp: definition.label output (numpy)."""
    valid = comp["in_T"]
    fig = plt.figure(figsize=(13, 9.2), dpi=110)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1, 1])
    vnp = np.asarray(valid)
    for j, pl in enumerate(("axial", "coronal", "sagittal")):
        slice_panel(fig.add_subplot(gs[0, j]), pt, pl, P, vnp, f"{pl} slice through the GB centroid")
    fig.suptitle("GB green, liver brown, body grey; orange = GB long axis (a1, length L); top row: skin contacts of all "
                 "candidates (grey) and of poses in T_GB^LA (blue), projected onto the slice\nfans: bone near-white, "
                 f"gas/lung dark, acoustic shadow darkened. T = visible & aligned & contained & acoustic ({cfg_status})",
                 fontsize=9, x=0.01, ha="left")
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    fails = {k: int((~np.asarray(comp[k])).sum()) for k in COMPONENTS}
    ax.text(0, 1, f"{sid}\nGB {pt.gb_ml:.1f} ml, length {pt.length:.0f} mm, widths "
                  f"{pt.widths[0]:.0f}/{pt.widths[1]:.0f} mm\n\ncandidates {len(vnp)}, in T {vnp.sum()} "
                  f"({vnp.mean() * 100:.1f}%)\nGB shadowed (mean over poses\n  that cut it): "
                  f"{s['shadow_frac'][s['area_geo'] > 0].mean():.0%}\n\nfailing each component:\n" +
            "\n".join(f"  {k:12s} {v:5d}" for k, v in fails.items()), va="top", family="monospace", fontsize=8)
    disp = {k: s[f"{k}_geo"] for k in ("area", "center_deg", "clip_frac", "depth_mm", "long_frac")}
    disp.update(align_deg=s["align_deg"], shadow_frac=s["shadow_frac"])
    one = lambda i: {k: disp[k][i] for k in disp}
    vi = np.nonzero(vnp)[0]
    qual = disp["area"] * disp["long_frac"] * (1 - disp["shadow_frac"])
    picks = (list(vi[np.argsort(-qual[vi])][:2]) + list(rng.choice(vi, min(2, len(vi)), replace=False))) if len(vi) else []
    for j, i in enumerate(picks[:4]):
        Pi = {k: v[i:i + 1] for k, v in P.items()}
        fan_panel(fig.add_subplot(gs[1, j]), pt, Pi, X, Y, in_fan, polar, one(i), ("best in T" if j < 2 else "random in T") + f" #{i}")
    # near misses: exactly one failed component, a different one per panel
    nf = sum((~np.asarray(comp[k])).astype(int) for k in COMPONENTS)
    shown = []
    for k in COMPONENTS:
        idx = np.nonzero((nf == 1) & ~np.asarray(comp[k]))[0]
        if len(idx) and len(shown) < 4:
            shown.append((idx[np.argmax(qual[idx])], k))
    for j, (i, k) in enumerate(shown):
        Pi = {kk: v[i:i + 1] for kk, v in P.items()}
        fan_panel(fig.add_subplot(gs[2, j]), pt, Pi, X, Y, in_fan, polar, one(i), f"near miss #{i}: fails {k}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_index(df):
    rows = "\n".join(
        f"<tr><td><a href='{r.study}.png'>{r.study}</a></td><td>{r.split}</td><td>{int(r.n_T)}</td><td>{r.n_T / r.n_candidates * 100:.1f}</td>"
        f"<td>{r.gb_ml:.1f}</td><td>{r.gb_length_mm:.0f}</td><td>{html.escape(str(r.adiposity_tertile))}</td>"
        f"<td>{r.fat_frac:.2f}</td></tr><tr><td colspan=8><a href='{r.study}.png'><img src='{r.study}.png' width=900></a></td></tr>"
        for r in df.itertuples())
    open(os.path.join(QC, "index.html"), "w").write(
        "<!doctype html><meta charset=utf-8><title>GB long-axis targets</title>"
        "<style>body{font:13px sans-serif;margin:16px}td{padding:2px 8px}</style>"
        "<h2>T_GB^LA long-axis target: inspection (train / val only)</h2><table><tr><th>study</th><th>split</th><th>in T</th>"
        "<th>in T %</th><th>GB ml</th><th>GB length mm</th><th>adiposity</th><th>fat_frac</th></tr>" + rows + "</table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20, help="number of studies (random, seeded) for inspection")
    ap.add_argument("--studies", nargs="+")
    ap.add_argument("--all", action="store_true", help="every study of the split")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--figures_only", action="store_true", help="redraw figures from the stored data (no scoring)")
    ap.add_argument("--config", default=os.path.join(paths.CONFIGS, "target_la.json"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    figs = not args.no_figures and args.split not in ("test", "external", "external2", "amos", "amos2", "amos_all")   # never draw test / external studies
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(QC, exist_ok=True)
    table = load_table(args.split)
    if args.studies:
        sids = args.studies
    elif args.all:
        sids = list(table.index)
    else:
        sids = sorted(np.random.default_rng(0).choice(table.index, args.n, replace=False))
    X, Y, in_fan, edge, polar = fan_grid()
    rows, t0 = [], time.time()
    for k, sid in enumerate(sids):
        row = table.loc[sid]
        seed = int(sid[1:])
        pt = Patient(row)
        path = os.path.join(OUT, f"{sid}.npz")
        if args.figures_only:
            d = np.load(path)
            P = {key: torch.from_numpy(d[key]).cuda() for key in ("apex", "beam", "lateral", "skin", "target")}
            s = {key: d[key] for key in d.files}
        else:
            g = torch.Generator(device="cuda").manual_seed(seed)
            P = pt.candidates(N_POSES, g)
            s = {key: v.cpu().numpy() for key, v in score(pt, P, X, Y, in_fan, edge, polar).items()}
        comp = label(s, cfg, s["area_geo"].max())
        n_T = int(comp["in_T"].sum())
        if not args.figures_only:
            np.savez(path, **{key: v.cpu().numpy() for key, v in P.items()}, **s,
                     **{key: np.asarray(v) for key, v in comp.items()}, target_config=json.dumps(cfg),
                     gb_centroid=pt.c.cpu().numpy(), gb_axes=pt.axes.cpu().numpy(), gb_length=pt.length,
                     gb_widths=np.array(pt.widths))
            rows.append({"study": sid, "split": args.split, "n_candidates": len(comp["in_T"]), "n_T": n_T,
                         "gb_ml": pt.gb_ml, "gb_length_mm": pt.length, "gb_width1_mm": pt.widths[0],
                         "gb_width2_mm": pt.widths[1],
                         "gb_shadow_frac_mean": float(s["shadow_frac"][s["area_geo"] > 0].mean()), "gas_ml": pt.gas_ml,
                         "fat_frac": row["fat_frac"], "adiposity_tertile": row["adiposity_tertile"],
                         **{f"pass_{c}_pct": float(np.mean(comp[c]) * 100) for c in COMPONENTS}})
        if figs:
            figure(pt, sid, P, s, comp, X, Y, in_fan, polar, os.path.join(QC, f"{sid}.png"), np.random.default_rng(seed),
                   cfg["status"].split(" ")[0])
        print(f"[{k + 1}/{len(sids)}] {sid}: GB {pt.gb_ml:.1f} ml, L {pt.length:.0f} mm | in T {n_T}/{len(comp['in_T'])} "
              f"| {(time.time() - t0) / 60:.1f} min", flush=True)
    if args.figures_only:
        df = pd.read_csv(os.path.join(paths.WORK, "gb_target_summary.csv"))
        write_index(df[df.split.isin(["train", "val"])])
        print(f"done -> {QC}/index.html")
        return
    df = pd.DataFrame(rows)
    path = os.path.join(paths.WORK, "gb_target_summary.csv")
    with locked(path):
        if os.path.exists(path):   # keep earlier studies, replace the ones redone
            old = pd.read_csv(path)
            df = pd.concat([old[~old.study.isin(df.study)], df]).sort_values("study")
        df.to_csv(path, index=False)
    if figs:
        write_index(df[df.split.isin(["train", "val"])])
    print(f"done -> {OUT}, {QC}/index.html")


if __name__ == "__main__":
    main()
