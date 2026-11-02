"""GB-blind source trajectories: a subcostal liver sweep (FROZEN 2026-10-23; configs/source_traj.json).

Generation uses ONLY the skin (body mask), acoustic blockers (bone masks, lung/trachea, CT gas), the costal margin
(right ribs 7-12 + costal cartilages, target/approach.py) and the liver. The GB mask is never loaded during generation.

Start: skin contact high in the epigastrium / right subcostal region, below_margin_mm under the costal margin and
x_right_of_midline_mm right of the midline; beam = inward skin normal tilted cephalad (under the ribs); image plane
approximately longitudinal. The start must satisfy the step constraints.
Step t -> t+1: a local search over n neighbours of P_t, each a random combination of in-plane slide + sweep slide
(re-seated on the skin), fan, rock and rotation within the neighbourhood magnitudes. Hard constraints:
  contact    share of the fan's rays that enter the body within contact_depth_mm of the probe face >= contact_min
             (per ray: the convex face is an arc, so its edges sit ~8 mm above flat skin; a real face compresses it)
  liver      unshadowed liver area in the fan >= liver_min_mm2
  shadow     shadowed share of the in-body fan <= shadow_max
  approach   inside A_subcostal (frozen target approach) and no more than sweep_below_margin_mm below the margin
  progress   lateral progress >= progress_min_mm (arc length around the vertical body axis, towards patient right)
Objective (no GB term): J = liver / liver_ref - l_s * shadow - l_m * d(P, P_t) / 10 + l_p * progress / 5 mm,
d = plane-invariant pose distance (skin mm + rotation deg modulo the 180 deg flip about the beam); the next pose
is sampled from softmax(J / temperature) over the feasible neighbours. Stop at max_steps or when no neighbour is
feasible (end of the approach laterally, or no usable liver window).

Post hoc (GB loaded only after generation):
  t_GB  censoring time = first pose (0-based) where the frozen target visibility criterion V holds
        (configs/target_la.json: geometric GB area >= max(0.5 A_max, 100 mm^2), GB depth 30-150 mm).
        The source sequence is O_1:t_GB = poses 0..t_GB-1: pre-target-acquisition observations (a sub-threshold
        GB sliver may already be in the image). t_GB = -1: V never holds, the whole trajectory is pre-acquisition.
  t_glimpse  first pose with >= gb_glimpse_mm2 unshadowed GB (reported only);
  membership in T_subcostal-LA (frozen labels + approach); distance to the nearest sampled pose of T_subcostal-LA;
  cumulative liver coverage (share of liver-surface points within 2 mm of an image plane so far).
Observations (GB-blind): per pose a 3 x 128 x 128 image on the sonobelief/fan.py grid: liver in fan (cleaned),
body in fan (FOV), acoustic shadow in fan; bit-packed (obs, along the last axis).
Output: work/source_traj/<study>.npz (+ <study>.csv rows), work/source_traj_summary.csv (--summarise),
        work/qc/source_traj/<study>.png (train/val with --figures only)
Training augmentation (train only, generator unchanged): --augment N draws N more trajectories per study with
seeds j = 8 .. 8+N-1 (the official set uses j = 0..7; seed = study * 100 + j) -> work/source_traj_aug/<study>.npz/.csv.
These are extra acquisitions of the same 302 train patients, used only to train the placement CFM; they are not
additional patients and never enter the summary or any val/test statistic.
Usage: python -m sonobelief.simulate.trajectories --split train [--all] [--shard i/n] [--studies ...] [--figures]
       python -m sonobelief.simulate.trajectories --split train --all --augment 64 [--shard i/n]
       python -m sonobelief.simulate.trajectories --summarise      (merge per-study rows; outcome statistics train/val only)
"""
import argparse
import json
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
from scipy import ndimage

from sonobelief import fan as G
from sonobelief import paths
from sonobelief.geometry import plane_rot_dist
from sonobelief.observation import clean_observation
from sonobelief.table import load_table
from sonobelief.target.approach import BIN_MM, landmarks, margin_profile
from sonobelief.target.definition import label, load_config
from sonobelief.target.perturbation import outward_normal, rotate, to_skin
from sonobelief.target.ruq_bank import (AIR_MASKS, BLK_AIR, BLK_BONE, BONE_PREFIXES, CMAP, GAS_HU, LAB_BODY, LAB_LIVER,
                                        N_RAYS, SEG_BODY, Patient, fan_grid, score)
from sonobelief.target.subcostal_bank import in_approach

matplotlib.use("Agg")


CFG_PATH = os.path.join(paths.CONFIGS, "source_traj.json")
OUT = os.path.join(paths.WORK, "source_traj")
OUT_AUG = os.path.join(paths.WORK, "source_traj_aug")
QC = os.path.join(paths.QC, "source_traj")


class BlindPatient(Patient):
    """Body / liver / acoustic blockers only. The gallbladder mask is NOT loaded (GB-blind generation)."""

    def __init__(self, row):
        seg = lambda n: nib.as_closest_canonical(nib.load(os.path.join(row["seg_dir"], n)))
        liver_img = seg("liver.nii.gz")
        liver = np.asanyarray(liver_img.dataobj) > 0
        body = np.asanyarray(nib.as_closest_canonical(nib.load(os.path.join(SEG_BODY, f"{row.name}.nii.gz"))).dataobj) > 0
        lab = body.astype(np.uint8) * LAB_BODY
        lab[liver] = LAB_LIVER
        self.sp = torch.tensor(liver_img.header.get_zooms()[:3], device="cuda", dtype=torch.float32)
        self.shape = torch.tensor(lab.shape, device="cuda", dtype=torch.float32)
        self.lab = torch.from_numpy(lab).cuda().float()[None, None]
        names = os.listdir(row["seg_dir"])
        assert not any("gallbladder" in n for n in [n for n in names if n.startswith(BONE_PREFIXES + AIR_MASKS)])
        load = lambda n: np.asanyarray(seg(n).dataobj) > 0
        with ThreadPoolExecutor(8) as ex:
            bone = list(ex.map(load, [n for n in names if n.startswith(BONE_PREFIXES)]))
            air = list(ex.map(load, [n for n in names if n.startswith(AIR_MASKS)]))
        ct = np.asanyarray(nib.as_closest_canonical(nib.load(row["ct"])).dataobj)
        blk = np.zeros(lab.shape, np.uint8)
        blk[np.logical_or.reduce(air) | ((ct < GAS_HU) & body)] = BLK_AIR
        blk[np.logical_or.reduce(bone)] = BLK_BONE
        self.blk = torch.from_numpy(blk).cuda().float()[None, None]
        bodyf = torch.from_numpy(body).cuda().float()[None, None]
        self.body = bodyf
        self.smooth = F.avg_pool3d(bodyf, 7, 1, 3)
        sp = np.array(self.sp.cpu())
        surf = np.argwhere(liver & ~ndimage.binary_erosion(liver)) * sp
        rng = np.random.default_rng(0)
        self.liver_surf = torch.from_numpy(surf[rng.choice(len(surf), min(6000, len(surf)), replace=False)]).cuda().float()


class Grid:
    def __init__(self, px, contact_mm, n_rays=48):
        self.X, self.Y, self.in_fan, self.edge, self.polar = fan_grid(px)
        self.px = px
        self.th = torch.linspace(-G.HALF, G.HALF, n_rays, device="cuda")
        self.r_near = G.R_APEX_MM + torch.arange(0, contact_mm + 0.5, 1.0, device="cuda")


def contact(pt, P, grid):
    """Share of rays entering the body within contact_depth_mm of the probe face."""
    d = torch.sin(grid.th)[:, None, None] * P["lateral"][None] + torch.cos(grid.th)[:, None, None] * P["beam"][None]
    pts = P["apex"][None, :, None] + grid.r_near[None, None, :, None] * d[:, :, None]       # (rays, B, R, 3)
    return (pt.sample(pt.body, pts) > 0.5).any(-1).float().mean(0)


def frame_metrics(pt, P, grid, chunk=64):
    """liver_mm2 (unshadowed liver in the fan), shadow (shadowed share of the in-body fan), contact."""
    out = {"liver_mm2": [], "shadow": [], "contact": []}
    for i in range(0, len(P["apex"]), chunk):
        Pc = {k: v[i:i + chunk] for k, v in P.items()}
        lab = pt.render(Pc, grid.X, grid.Y)
        sh = pt.shadow(Pc, grid.polar)
        inb = (lab > 0) & grid.in_fan
        out["liver_mm2"].append(((lab == LAB_LIVER) & grid.in_fan & ~sh).sum((1, 2)).float() * grid.px ** 2)
        out["shadow"].append((inb & sh).sum((1, 2)).float() / inb.sum((1, 2)).clamp(min=1))
        out["contact"].append(contact(pt, Pc, grid))
    return {k: torch.cat(v) for k, v in out.items()}


def frames(P):
    lat, beam = P["lateral"], P["beam"]
    return torch.stack([lat, beam, torch.cross(lat, beam, dim=-1)], -1)


def margin_z(x, edges, zcm):
    idx = ((x - float(edges[0])) / BIN_MM).long().clamp(0, len(zcm) - 1)
    return torch.from_numpy(np.asarray(zcm)).cuda().float()[idx]


class Sweep:
    def __init__(self, pt, lm, cfg, cfg_t, grid):
        self.pt, self.cfg, self.cfg_t, self.grid = pt, cfg, cfg_t, grid
        self.mid_x, self.spine_y, self.edges, self.zcm = lm
        self.ap = cfg_t["approach"]

    def psi(self, skin):
        """Azimuth of the skin contact around the vertical axis through the spine (rad; + = towards patient right)."""
        return torch.atan2(skin[:, 0] - self.mid_x, skin[:, 1] - self.spine_y)

    def feasible(self, P, m):
        c = self.cfg["constraints"]
        ok = torch.isfinite(P["skin"]).all(1)
        ok &= m["contact"] >= c["contact_min"]
        ok &= m["liver_mm2"] >= c["liver_min_mm2"]
        ok &= m["shadow"] <= c["shadow_max"]
        Pn = {k: torch.nan_to_num(v) for k, v in P.items()}
        ok &= in_approach(self.pt, Pn["skin"], Pn["beam"], self.mid_x, self.edges, self.zcm, self.ap)
        zc = margin_z(Pn["skin"][:, 0], self.edges, self.zcm)
        ok &= Pn["skin"][:, 2] >= zc - c["sweep_below_margin_mm"]
        return ok

    def start(self, gen):
        s = self.cfg["start"]
        rnd = lambda lo, hi, n: lo + (hi - lo) * torch.rand(n, device="cuda", generator=gen)
        n = s["tries"]
        # medial reference: the midline, or the most medial defined costal-margin bin if the margin starts further right
        # (medial arch outside the CT field of view / not segmented); unchanged for studies with a medial margin
        zcm = np.asarray(self.zcm, float)
        x_first = float(self.edges[0]) + BIN_MM * int(np.argmax(np.isfinite(zcm)))
        x0 = max(float(self.mid_x), x_first) + rnd(*s["x_right_of_midline_mm"], n)
        z0 = margin_z(x0, self.edges, self.zcm) - rnd(*s["below_margin_mm"], n)
        # anterior skin: march forward from the spine plane at (x0, z0)
        ys = torch.arange(0, 400, 1.0, device="cuda")
        pts = torch.stack([x0[:, None].expand(-1, len(ys)), self.spine_y + ys[None].expand(n, -1), z0[:, None].expand(-1, len(ys))], -1)
        inside = self.pt.sample(self.pt.body, pts) > 0.5
        last = (inside.int() * torch.arange(len(ys), device="cuda")[None]).argmax(1)
        skin = pts[torch.arange(n), last]
        inward = -outward_normal(self.pt, skin)
        zhat = torch.tensor([0.0, 0.0, 1.0], device="cuda").expand_as(inward)
        zp = F.normalize(zhat - (zhat * inward).sum(-1, keepdim=True) * inward, dim=-1)
        a = torch.deg2rad(rnd(*s["cephalad_tilt_deg"], n))
        beam = F.normalize(torch.cos(a)[:, None] * inward + torch.sin(a)[:, None] * zp, dim=-1)
        theta = torch.deg2rad(rnd(-s["plane_from_longitudinal_deg"], s["plane_from_longitudinal_deg"], n))
        lateral = G.lateral_axis(beam, theta)
        P = {"skin": skin, "beam": beam, "lateral": lateral, "apex": skin - G.R_APEX_MM * beam}
        m = frame_metrics(self.pt, P, self.grid)
        ok = self.feasible(P, m) & inside.any(1)
        if not ok.any():
            return None
        o = self.cfg["objective"]   # best feasible start (liver window, little shadow); no GB term
        J = torch.where(ok, m["liver_mm2"] / o["liver_ref_mm2"] - o["lambda_shadow"] * m["shadow"], torch.tensor(-1e9, device="cuda"))
        i = int(torch.multinomial(torch.softmax(J / o["temperature"], 0), 1, generator=gen))
        return {k: v[i:i + 1] for k, v in P.items()}

    def neighbours(self, P, gen):
        nb = self.cfg["neighbourhood"]
        n = nb["n"]
        u = lambda a: (2 * torch.rand(n, device="cuda", generator=gen) - 1) * a
        skin, beam, lat = (P[k].expand(n, -1) for k in ("skin", "beam", "lateral"))
        normal = torch.cross(lat, beam, dim=-1)
        n_out = outward_normal(self.pt, skin)
        tang = lambda v: F.normalize(v - (v * n_out).sum(-1, keepdim=True) * n_out, dim=-1)
        shift = u(nb["slide_mm"])[:, None] * tang(lat) + u(nb["slide_mm"])[:, None] * tang(normal)
        skin = to_skin(self.pt, skin + shift, n_out)
        rock, fan, rot = (torch.deg2rad(u(nb[k])) for k in ("rock_deg", "fan_deg", "rotate_deg"))
        beam, lat = rotate(beam, normal, rock), rotate(lat, normal, rock)
        beam = rotate(beam, lat, fan)
        lat = rotate(lat, beam, rot)
        lat = F.normalize(lat - (lat * beam).sum(-1, keepdim=True) * beam, dim=-1)
        return {"skin": skin, "beam": beam, "lateral": lat, "apex": skin - G.R_APEX_MM * beam}

    def run(self, gen):
        o, c = self.cfg["objective"], self.cfg["constraints"]
        P = self.start(gen)
        if P is None:
            return None
        poses, mets = [P], [frame_metrics(self.pt, P, self.grid)]
        for _ in range(self.cfg["max_steps"] - 1):
            Q = self.neighbours(P, gen)
            m = frame_metrics(self.pt, {k: torch.nan_to_num(v) for k, v in Q.items()}, self.grid)
            ok = self.feasible(Q, m)
            Qn = {k: torch.nan_to_num(v) for k, v in Q.items()}
            r0 = torch.hypot(P["skin"][:, 0] - self.mid_x, P["skin"][:, 1] - self.spine_y)
            prog = (self.psi(Qn["skin"]) - self.psi(P["skin"])) * r0
            ok &= prog >= c["progress_min_mm"]
            if not ok.any():
                break
            d = (Qn["skin"] - P["skin"]).norm(dim=1) + plane_rot_dist(frames(P)[0], frames(Qn))
            J = (m["liver_mm2"] / o["liver_ref_mm2"] - o["lambda_shadow"] * m["shadow"] - o["lambda_motion"] * d / 10
                 + o["lambda_progress"] * prog / 5)
            J = torch.where(ok, J, torch.tensor(-1e9, device="cuda"))
            i = int(torch.multinomial(torch.softmax(J / o["temperature"], 0), 1, generator=gen))
            P = {k: v[i:i + 1] for k, v in Qn.items()}
            poses.append(P)
            mets.append({k: v[i:i + 1] for k, v in m.items()})
        cat = lambda L, k: torch.cat([x[k] for x in L])
        T = {k: cat(poses, k) for k in ("skin", "beam", "lateral", "apex")}
        M = {k: cat(mets, k) for k in ("liver_mm2", "shadow", "contact")}
        T["psi_deg"] = torch.rad2deg(self.psi(T["skin"]))
        T["z_rel_margin"] = T["skin"][:, 2] - margin_z(T["skin"][:, 0], self.edges, self.zcm)
        return T, M

    def coverage(self, T, slab=2.0):
        """Cumulative share of liver-surface points within slab mm of an image plane (inside the fan) up to step t."""
        L = self.pt.liver_surf
        seen = torch.zeros(len(L), dtype=torch.bool, device="cuda")
        out = []
        for t in range(len(T["apex"])):
            a, b, l = T["apex"][t], T["beam"][t], T["lateral"][t]
            v = L - a
            y, x, z = v @ b, v @ l, v @ torch.cross(l, b, dim=0)
            r, ang = torch.sqrt(x ** 2 + y ** 2), torch.atan2(x, y).abs()
            seen |= (z.abs() <= slab) & (r >= G.R_APEX_MM) & (r <= G.R_MAX) & (ang <= G.HALF)
            out.append(seen.float().mean())
        return torch.stack(out)


def posthoc(row, sid, trajs, cfg, cfg_t, lm, grid1):
    """GB-dependent analysis AFTER generation: GB visibility, first sighting, membership / distance to T."""
    gpt = Patient(row)
    dense = np.load(os.path.join(paths.WORK, "gb_target_subcostal", f"{sid}.npz"))
    ti = np.nonzero(dense["in_T"])[0]
    TP = {k: torch.from_numpy(dense[k][ti]).cuda() for k in ("skin", "beam", "lateral")}
    RT = torch.stack([TP["lateral"], TP["beam"], torch.cross(TP["lateral"], TP["beam"], dim=-1)], -1)
    mid_x, spine_y, edges, zcm = lm
    res = []
    for T in trajs:
        P = {k: T[k] for k in ("apex", "beam", "lateral", "skin")}
        s = {k: v.cpu().numpy() for k, v in score(gpt, P, grid1.X, grid1.Y, grid1.in_fan, grid1.edge, grid1.polar).items()}
        lab = label(s, cfg_t, float(dense["area_max"]))
        appr = in_approach(gpt, P["skin"], P["beam"], mid_x, edges, zcm, cfg_t["approach"]).cpu().numpy()
        in_T = lab["in_T"] & appr
        first = lambda m: int(np.argmax(m)) if m.any() else -1
        t_gb = first(lab["visible"])                        # censoring: frozen target visibility criterion V
        t_alt = {"t_glimpse": first(s["area"] >= cfg["posthoc"]["gb_glimpse_mm2"])}
        if len(ti):
            R = frames(P)
            dist = torch.stack([((TP["skin"] - P["skin"][t]).norm(dim=1) + plane_rot_dist(R[t], RT)).min()
                                for t in range(len(R))]).cpu().numpy()
        else:
            dist = np.full(len(s["area"]), np.nan)
        res.append({"gb_area_vis": s["area"], "gb_area_geo": s["area_geo"], "t_gb": t_gb, "in_T": in_T, "dist_to_T": dist, **t_alt})
    return res, gpt


def figure(sid, bpt, gpt, trajs, mets, ph, lm, path, grid_img):
    mid_x, spine_y, edges, zcm = lm
    fig = plt.figure(figsize=(16, 8.8), dpi=100)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, :2])
    xc = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    ax.plot(xc - mid_x, np.asarray(zcm), color="#8a8a85", lw=2, label="costal margin (lowest arch point per x)")
    cols = plt.cm.tab10(np.arange(len(trajs)))
    for j, (T, r) in enumerate(zip(trajs, ph)):
        sk = T["skin"].cpu().numpy()
        ax.plot(sk[:, 0] - mid_x, sk[:, 2], color=cols[j], lw=1.4)
        ax.scatter(sk[0, 0] - mid_x, sk[0, 2], color=cols[j], s=25, marker="o")
        if r["t_gb"] >= 0:
            ax.scatter(sk[r["t_gb"], 0] - mid_x, sk[r["t_gb"], 2], color=cols[j], s=60, marker="x")
        if r["in_T"].any():
            k = int(np.argmax(r["in_T"]))
            ax.scatter(sk[k, 0] - mid_x, sk[k, 2], color=cols[j], s=70, marker="*", edgecolor="k", lw=0.5)
    gc = gpt.c.cpu().numpy()
    ax.scatter(gc[0] - mid_x, gc[2], s=120, marker="P", color="#1f9a52", label="GB centroid (post hoc)")
    ax.set_xlabel("x from midline (mm, + = patient right)")
    ax.set_ylabel("z (mm, + = head)")
    ax.set_title(f"{sid}: {len(trajs)} skin paths\n(o start, x censoring t_GB = first pose meeting V, * first pose in T; post hoc)", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    ax.set_aspect("equal")
    ax2 = fig.add_subplot(gs[0, 2:4])
    for j, (m, r) in enumerate(zip(mets, ph)):
        ax2.plot(m["liver_mm2"].cpu().numpy() / 100, color=cols[j], lw=1)
    ax2.set_ylabel("unshadowed liver in view (cm²)")
    ax2.set_xlabel("step")
    ax2.set_title("liver window along the sweep", fontsize=9, loc="left")
    ax3 = fig.add_subplot(gs[0, 4:6])
    for j, r in enumerate(ph):
        ax3.plot(r["dist_to_T"], color=cols[j], lw=1)
        if r["t_gb"] >= 0:
            ax3.scatter(r["t_gb"], r["dist_to_T"][r["t_gb"]], color=cols[j], marker="x")
    ax3.set_ylabel("distance to nearest target pose (mm + deg)")
    ax3.set_xlabel("step")
    ax3.set_title("post hoc: distance to T_subcostal-LA", fontsize=9, loc="left")
    # frames of trajectory 0: 6 evenly spaced up to the end, GB-first frame flagged
    T, r = trajs[0], ph[0]
    n = len(T["skin"])
    idx = sorted(set(np.linspace(0, n - 1, 6).round().astype(int).tolist()))
    for k, t in enumerate(idx[:6]):
        axf = fig.add_subplot(gs[1, k])
        Pi = {kk: T[kk][t:t + 1] for kk in ("apex", "beam", "lateral")}
        lab = bpt.render(Pi, grid_img.X, grid_img.Y)[0].cpu().numpy()
        blk = bpt.render(Pi, grid_img.X, grid_img.Y, bpt.blk)[0].cpu().numpy()
        sh = bpt.shadow(Pi, grid_img.polar)[0].cpu().numpy()
        rgb = CMAP[lab].copy()
        rgb[sh] *= 0.45
        rgb[blk == BLK_BONE] = (0.98, 0.95, 0.80)
        rgb[(blk == BLK_AIR) & (lab > 0)] = (0.15, 0.15, 0.18)
        rgb[~grid_img.in_fan.cpu().numpy()] = 1.0
        axf.imshow(rgb, extent=[grid_img.X.min().item(), grid_img.X.max().item(), grid_img.Y.max().item(), grid_img.Y.min().item()])
        axf.set_xticks([]); axf.set_yticks([])
        flag = " (censored: GB acquired, V)" if 0 <= r["t_gb"] <= t else ""
        axf.set_title(f"traj 1, step {t + 1}/{n}{flag}", fontsize=8, loc="left", color="#c0392b" if flag else "k")
    fig.suptitle("GB-blind subcostal sweep (frames: what the model observes - liver brown, body grey, bone white, gas dark, "
                 "shadow darkened; the GB is not rendered)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


class ObsGrid:
    """sonobelief/fan.py observation grid (IMG x IMG, PX_MM) with the polar lookup for shadows."""

    def __init__(self):
        self.X, self.Y, self.in_fan = G.pixel_grid("cuda")
        r, ang = torch.sqrt(self.X ** 2 + self.Y ** 2), torch.atan2(self.X, self.Y)
        ia = ((ang.clamp(-G.HALF, G.HALF) + G.HALF) / (2 * G.HALF) * (N_RAYS - 1)).round().long()
        ir = (r - G.R_APEX_MM).round().long().clamp(0, int(G.DEPTH_MM))
        self.polar = (ia, ir)


def render_obs(pt, T, og, chunk=64):
    """GB-blind observations (n, 3, IMG, IMG) bool: liver in fan (cleaned), body in fan, shadow in fan."""
    out = []
    for i in range(0, len(T["apex"]), chunk):
        P = {k: T[k][i:i + chunk] for k in ("apex", "beam", "lateral")}
        lab = pt.render(P, og.X, og.Y)
        sh = pt.shadow(P, og.polar)
        out.append(torch.stack([(lab == LAB_LIVER) & og.in_fan, (lab > 0) & og.in_fan, sh & og.in_fan], 1))
    o = torch.cat(out).cpu().numpy()
    o[:, :2] = clean_observation(o[:, :2])
    return o


def study_rows(sid, split, trajs, mets, covs, ph):
    rows = []
    for j, (T, M, c, r) in enumerate(zip(trajs, mets, covs, ph)):
        sk = T["skin"].cpu().numpy()
        R = frames(T)
        step_mm = np.linalg.norm(np.diff(sk, axis=0), axis=1)
        step_deg = np.array([plane_rot_dist(R[t], R[t + 1:t + 2]).item() for t in range(len(sk) - 1)])
        n = len(sk)
        rows.append({"study": sid, "split": split, "traj": j, "n_steps": n,
                     "step_mm_med": np.median(step_mm) if n > 1 else np.nan,
                     "step_mm_p95": np.percentile(step_mm, 95) if n > 1 else np.nan,
                     "step_deg_med": np.median(step_deg) if n > 1 else np.nan,
                     "step_deg_p95": np.percentile(step_deg, 95) if n > 1 else np.nan,
                     "liver_cm2_med": float(M["liver_mm2"].median()) / 100, "shadow_med": float(M["shadow"].median()),
                     "psi_start": float(T["psi_deg"][0]), "psi_end": float(T["psi_deg"][-1]),
                     "path_mm": float(step_mm.sum()), "z_rel_margin_med": float(T["z_rel_margin"].median()),
                     "t_gb": r["t_gb"], "gb_acquired": r["t_gb"] >= 0, "n_pre": r["t_gb"] if r["t_gb"] >= 0 else n,
                     "t_glimpse": r["t_glimpse"], "n_pre_glimpse": r["t_glimpse"] if r["t_glimpse"] >= 0 else n,
                     "reaches_T": bool(r["in_T"].any()), "first_T": int(np.argmax(r["in_T"])) if r["in_T"].any() else -1,
                     "dist_T_start": float(r["dist_to_T"][0]), "dist_T_min": float(np.nanmin(r["dist_to_T"])) if np.isfinite(r["dist_to_T"]).any() else np.nan,
                     "coverage_end": float(c[-1]),
                     "coverage_pre": float(c[r["t_gb"] - 1]) if r["t_gb"] > 0 else (float(c[-1]) if r["t_gb"] < 0 else 0.0)})
    # diversity: mean (over steps) distance from each trajectory to the nearest pose of each other trajectory
    div = []
    for a in range(len(trajs)):
        for b in range(len(trajs)):
            if a != b:
                Ra, Rb = frames(trajs[a]), frames(trajs[b])
                dd = torch.stack([((trajs[b]["skin"] - trajs[a]["skin"][t]).norm(dim=1) + plane_rot_dist(Ra[t], Rb)).min()
                                  for t in range(len(Ra))])
                div.append(dd.mean().item())
    for rr in rows:
        rr["diversity_med"] = float(np.median(div)) if div else np.nan
    return rows


def summarise():
    """Merge per-study rows; generation counts for every split, outcome statistics for train/val only."""
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".csv"))
    df = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in files], ignore_index=True)
    labs = pd.read_csv(os.path.join(paths.WORK, "gb_target_summary.csv"))[["study", "applicable", "available", "difficult"]]
    df = df.drop(columns=[c for c in ("applicable", "available", "difficult") if c in df]).merge(labs, on="study", how="left")
    df.to_csv(os.path.join(paths.WORK, "source_traj_summary.csv"), index=False)
    print("generated (studies / trajectories / studies without a feasible start):")
    for sp, g in df.groupby("split"):
        print(f"  {sp:5s} {g.study.nunique()} / {int(g.traj.notna().sum())} / {int(g.traj.isna().groupby(g.study).all().sum())}")
    tv = df[df.split.isin(["train", "val"]) & df.traj.notna()]
    if not len(tv):
        return
    q = lambda c: f"{tv[c].median():.2f} (IQR {tv[c].quantile(0.25):.2f}-{tv[c].quantile(0.75):.2f})"
    print(f"\ntrain/val outcome statistics: {tv.study.nunique()} studies, {len(tv)} trajectories")
    for c in ("n_steps", "step_mm_med", "step_mm_p95", "step_deg_med", "step_deg_p95", "liver_cm2_med", "shadow_med",
              "path_mm", "z_rel_margin_med", "n_pre", "n_pre_glimpse", "dist_T_start", "dist_T_min",
              "coverage_pre", "coverage_end", "diversity_med"):
        print(f"  {c:18s} {q(c)}")
    for c in ("n_pre", "n_pre_glimpse"):
        print(f"  {c}: share of trajectories with >= 1 / 2 / 4 / 8 / 16 views: "
              + " / ".join(f"{(tv[c] >= k).mean():.0%}" for k in (1, 2, 4, 8, 16)))
    main_set = tv[tv.applicable.astype(bool) & tv.available.astype(bool)]
    print(f"  applicable & available: {main_set.study.nunique()} studies, {len(main_set)} trajectories; >= 8 views "
          f"{(main_set.n_pre >= 8).mean():.0%}, >= 16 {(main_set.n_pre >= 16).mean():.0%}")
    print(f"  GB acquired (V) in {tv.gb_acquired.mean():.0%}; reaches T in {tv.reaches_T.mean():.0%}; V at the first pose in {(tv.t_gb == 0).mean():.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--all", action="store_true", help="every study of the split (default: --n random studies)")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--studies", nargs="+")
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles studies i, i+n, ...")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--figures", action="store_true", help="QC figures (train/val only)")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--augment", type=int, default=0, help="train-only augmentation: N extra trajectories per study")
    args = ap.parse_args()
    if args.summarise:
        return summarise()
    cfg, cfg_t = json.load(open(CFG_PATH)), load_config()
    if args.split in ("test", "external", "external2", "amos", "amos2", "amos_all"):
        assert cfg["status"].startswith("frozen"), "test / external trajectories only with the frozen generator"
        assert not args.figures, "no QC figures on test / external"
    if args.augment:
        assert args.split == "train" and not args.figures, "augmentation is train-only, without figures"
        n0 = cfg["n_traj_per_study"]
        assert n0 + args.augment <= 100, "seed = study * 100 + j needs j < 100"
        out_dir, js = OUT_AUG, list(range(n0, n0 + args.augment))
    else:
        out_dir, js = OUT, list(range(cfg["n_traj_per_study"]))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(QC, exist_ok=True)
    table = load_table(args.split)
    if args.studies:
        sids = args.studies
    elif args.all:
        sids = sorted(table.index)          # generation is GB-blind: every study; labels are attached in the summary
    else:
        summ = pd.read_csv(os.path.join(paths.WORK, "gb_target_summary.csv")).set_index("study")
        pool = [s for s in table.index if bool(summ.loc[s, "applicable"]) and bool(summ.loc[s, "available"])]
        sids = sorted(np.random.default_rng(3).choice(pool, min(args.n, len(pool)), replace=False))
    si, sn = map(int, args.shard.split("/"))
    sids = sids[si::sn]
    grid = Grid(cfg["search_px_mm"], cfg["constraints"]["contact_depth_mm"])
    grid1 = Grid(1.0, cfg["constraints"]["contact_depth_mm"])
    og = ObsGrid()
    t0 = time.time()
    for k, sid in enumerate(sids):
        npz, csv = os.path.join(out_dir, f"{sid}.npz"), os.path.join(out_dir, f"{sid}.csv")
        if os.path.exists(csv) and not args.overwrite:
            continue
        row = table.loc[sid]
        bpt = BlindPatient(row)                                   # GB-blind
        costal, sp, mid_x, spine_y = landmarks(row)
        edges, zcm = margin_profile(costal, sp, mid_x, spine_y, cfg_t["approach"])
        lm = (mid_x, spine_y, edges, zcm)
        sw = Sweep(bpt, lm, cfg, cfg_t, grid)
        trajs, mets, covs, seeds = [], [], [], []
        for j in js:
            g = torch.Generator(device="cuda").manual_seed(int(sid[1:]) * 100 + j)
            out = sw.run(g)
            if out is None:
                continue
            T, M = out
            trajs.append(T); mets.append(M); covs.append(sw.coverage(T)); seeds.append(j)
        if not trajs:
            pd.DataFrame([{"study": sid, "split": args.split}]).to_csv(csv, index=False)
            print(f"[{k + 1}/{len(sids)}] {sid}: no feasible start", flush=True)
            continue
        lengths = np.array([len(T["skin"]) for T in trajs])
        obs = np.packbits(np.concatenate([render_obs(bpt, T, og) for T in trajs]), axis=-1)   # GB-blind, before post hoc
        ph, gpt = posthoc(row, sid, trajs, cfg, cfg_t, lm, grid1)  # GB used only from here on
        L = cfg["max_steps"]
        pad = lambda arrs: np.stack([np.pad(a, [(0, L - len(a))] + [(0, 0)] * (a.ndim - 1), constant_values=np.nan) for a in arrs])
        np.savez_compressed(
            npz, length=lengths, offset=np.concatenate([[0], np.cumsum(lengths)[:-1]]), obs=obs,
            **{key: pad([T[key].cpu().numpy() for T in trajs]) for key in ("skin", "beam", "lateral", "apex", "psi_deg", "z_rel_margin")},
            **{key: pad([M[key].cpu().numpy() for M in mets]) for key in ("liver_mm2", "shadow", "contact")},
            coverage=pad([c.cpu().numpy() for c in covs]),
            t_gb=np.array([r["t_gb"] for r in ph]), t_glimpse=np.array([r["t_glimpse"] for r in ph]),
            gb_area_vis=pad([r["gb_area_vis"] for r in ph]), gb_area_geo=pad([r["gb_area_geo"] for r in ph]),
            in_T=pad([r["in_T"].astype(float) for r in ph]), dist_to_T=pad([r["dist_to_T"] for r in ph]),
            seed_j=np.array(seeds), config=json.dumps(cfg))
        rows = study_rows(sid, args.split, trajs, mets, covs, ph)
        pd.DataFrame(rows).to_csv(csv, index=False)
        if args.figures:
            figure(sid, bpt, gpt, trajs, mets, ph, lm, os.path.join(QC, f"{sid}.png"), grid1)
        msg = f"[{k + 1}/{len(sids)}] {sid}: {len(trajs)} traj, {lengths.sum()} poses"
        if args.split not in ("test", "external", "external2", "amos", "amos2", "amos_all"):
            msg += f", steps {lengths.tolist()}, t_GB {[r['t_gb'] for r in ph]}"
        print(msg + f" | {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
