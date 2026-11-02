"""Dense sampling of T_subcostal-LA = T_LA ∩ A_subcostal (first target family).

A_subcostal (target/approach.py, configs/target_la.json "approach"): our operationalisation of the clinically described
right-subcostal approach (not a clinical standard): skin contact from band_above_mm above to band_below_mm below
the patient's right costal margin, from the epigastrium (x >= midline - max_left_mm) to about the anterior axillary
line (skin-normal azimuth in normal_azimuth_deg); beam >= 30 deg posterior and <= 20 deg caudal.

Candidates (N_POSES per study, seeded): skin contacts drawn uniformly from the body-surface voxels inside the
approach region; beam towards the GB centroid + N(0, AIM_JITTER_MM^2 I), kept only if it satisfies the approach
beam rule, is within MAX_NORMAL_DEG of the inward skin normal, reaches the target within the imaging depth and
enters the body; roll uniform. Every candidate is in A_subcostal by construction, so in_T here = T_subcostal-LA.
Rendering, shadowing, measurements and the frozen T_LA labels are those of target/ruq_bank.py / target/definition.py.
Output: work/gb_target_subcostal/<study>.npz (as gb_target), work/gb_target_subcostal_summary.csv
Usage: python -m sonobelief.target.subcostal_bank [--split train] [--studies ...]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sonobelief import fan as G
from sonobelief import paths
from sonobelief.table import load_table
from sonobelief.target.approach import BIN_MM, landmarks, margin_profile
from sonobelief.target.definition import CONFIG, label, load_config
from sonobelief.target.locked import locked
from sonobelief.target.ruq_bank import AIM_JITTER_MM, MAX_NORMAL_DEG, N_POSES, Patient, fan_grid, score

OUT = os.path.join(paths.WORK, "gb_target_subcostal")


def region_points(pt, mid_x, edges, zcm, ap):
    """Body-surface voxels (mm) inside the approach region, and their inward normals."""
    s = pt.surface
    s = s[s[:, 0] >= mid_x - ap["max_left_mm"]]
    idx = ((s[:, 0] - edges[0]) / BIN_MM).long()
    zc = torch.from_numpy(zcm).cuda().float()
    ok = (idx >= 0) & (idx < len(zcm))
    z = torch.full((len(s),), float("nan"), device="cuda")
    z[ok] = zc[idx[ok]]
    ok &= torch.isfinite(z) & (s[:, 2] >= z - ap["band_below_mm"]) & (s[:, 2] <= z + ap["band_above_mm"])
    s = s[ok]
    grad = torch.stack([pt.sample(pt.smooth, s + 3 * e, "bilinear") - pt.sample(pt.smooth, s - 3 * e, "bilinear")
                        for e in torch.eye(3, device="cuda")], -1)
    inward = F.normalize(grad, dim=-1)
    az = torch.rad2deg(torch.atan2(-inward[:, 0], -inward[:, 1]))     # azimuth of the OUTWARD normal
    keep = (grad.norm(dim=-1) > 1e-3) & (az >= ap["normal_azimuth_deg"][0]) & (az <= ap["normal_azimuth_deg"][1])
    return s[keep], inward[keep]


def in_approach(pt, skin, beam, mid_x, edges, zcm, ap):
    """A_subcostal membership of arbitrary poses (skin (n, 3), beam (n, 3), torch) -> bool (n,)."""
    idx = ((skin[:, 0] - float(edges[0])) / BIN_MM).long()
    zc = torch.from_numpy(np.asarray(zcm)).cuda().float()
    ok = (idx >= 0) & (idx < len(zc))
    z = torch.full((len(skin),), float("nan"), device="cuda")
    z[ok] = zc[idx[ok]]
    ok &= torch.isfinite(z) & (skin[:, 0] >= mid_x - ap["max_left_mm"])
    ok &= (skin[:, 2] >= z - ap["band_below_mm"]) & (skin[:, 2] <= z + ap["band_above_mm"])
    grad = torch.stack([pt.sample(pt.smooth, skin + 3 * e, "bilinear") - pt.sample(pt.smooth, skin - 3 * e, "bilinear")
                        for e in torch.eye(3, device="cuda")], -1)
    inward = F.normalize(grad, dim=-1)
    az = torch.rad2deg(torch.atan2(-inward[:, 0], -inward[:, 1]))
    ok &= (az >= ap["normal_azimuth_deg"][0]) & (az <= ap["normal_azimuth_deg"][1])
    ok &= (beam[:, 2] >= math.sin(math.radians(ap["min_cephalad_deg"])))
    ok &= (-beam[:, 1] >= math.cos(math.radians(ap["max_from_posterior_deg"])))
    return ok


def candidates(pt, skin_pts, inward_pts, ap, n, gen):
    rnd = lambda *sz: torch.rand(*sz, device="cuda", generator=gen)
    out, have = {k: [] for k in ("apex", "beam", "lateral", "skin", "target")}, 0
    sin_c, cos_p = math.sin(math.radians(ap["min_cephalad_deg"])), math.cos(math.radians(ap["max_from_posterior_deg"]))
    for _ in range(200):
        m = 4 * n
        i = torch.randint(len(skin_pts), (m,), device="cuda", generator=gen)
        skin = skin_pts[i] + (rnd(m, 3) - 0.5) * pt.sp
        inward = inward_pts[i]
        target = pt.c + AIM_JITTER_MM * torch.randn(m, 3, device="cuda", generator=gen)
        beam = target - skin
        dist = beam.norm(dim=1)
        beam = beam / dist[:, None]
        ok = ((beam * inward).sum(-1) >= math.cos(math.radians(MAX_NORMAL_DEG)))
        ok &= (beam[:, 2] >= sin_c) & (-beam[:, 1] >= cos_p)
        ok &= (dist < G.DEPTH_MM - 10) & (pt.sample(pt.body, skin + 5 * beam) > 0.5)
        lateral = G.lateral_axis(beam, rnd(m) * math.pi)
        for k, v in (("apex", skin - G.R_APEX_MM * beam), ("beam", beam), ("lateral", lateral), ("skin", skin), ("target", target)):
            out[k].append(v[ok])
        have += int(ok.sum())
        if have >= n:
            break
    return {k: torch.cat(v)[:n] for k, v in out.items()}, have


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--split", default="train")
    ap_.add_argument("--studies", nargs="+")
    args = ap_.parse_args()
    cfg = load_config(CONFIG)
    ap = cfg["approach"]
    os.makedirs(OUT, exist_ok=True)
    table = load_table(args.split)
    sids = args.studies or list(table.index)
    X, Y, in_fan, edge, polar = fan_grid()
    rows, t0 = [], time.time()
    for k, sid in enumerate(sids):
        row = table.loc[sid]
        pt = Patient(row)
        costal, sp, mid_x, spine_y = landmarks(row)
        edges, zcm = margin_profile(costal, sp, mid_x, spine_y, ap)
        skin_pts, inward_pts = region_points(pt, mid_x, edges, zcm, ap)
        g = torch.Generator(device="cuda").manual_seed(int(sid[1:]) + 7)
        n_got = 0
        if len(skin_pts):
            P, n_got = candidates(pt, skin_pts, inward_pts, ap, N_POSES, g)
        if n_got == 0:
            rows.append({"study": sid, "split": args.split, "n_region_skin": len(skin_pts), "n_candidates": 0, "n_T_sub": 0})
            print(f"[{k + 1}/{len(sids)}] {sid}: no feasible approach candidates", flush=True)
            continue
        s = {key: v.cpu().numpy() for key, v in score(pt, P, X, Y, in_fan, edge, polar).items()}
        # the relative-area reference stays the patient's largest GB cut over ALL approaches (gb_target), so that
        # T_subcostal-LA is exactly T_LA restricted to the approach
        ref = np.load(os.path.join(paths.WORK, "gb_target", f"{sid}.npz"))
        area_max = max(float(ref["area_geo"].max()), float(s["area_geo"].max()))
        comp = label(s, cfg, area_max)
        n_T = int(comp["in_T"].sum())
        np.savez(os.path.join(OUT, f"{sid}.npz"), **{key: v.cpu().numpy() for key, v in P.items()}, **s,
                 **{key: np.asarray(v) for key, v in comp.items()}, area_max=area_max, target_config=json.dumps(cfg),
                 gb_centroid=pt.c.cpu().numpy(), gb_axes=pt.axes.cpu().numpy(), gb_length=pt.length,
                 margin_edges=edges, margin_z=zcm, midline_x=mid_x)
        rows.append({"study": sid, "split": args.split, "n_region_skin": len(skin_pts), "n_candidates": len(comp["in_T"]),
                     "n_T_sub": n_T, **{f"pass_{c}_pct": float(np.mean(comp[c]) * 100) for c in ("visible", "aligned", "contained", "acoustic")}})
        print(f"[{k + 1}/{len(sids)}] {sid}: candidates {len(comp['in_T'])} | T_sub {n_T} | {(time.time() - t0) / 60:.1f} min",
              flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(paths.WORK, "gb_target_subcostal_summary.csv")
    with locked(path):
        if os.path.exists(path):
            old = pd.read_csv(path)
            df = pd.concat([old[~old.study.isin(df.study)], df]).sort_values("study")
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
