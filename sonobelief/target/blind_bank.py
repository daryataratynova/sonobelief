"""GB-blind candidate bank over A_subcostal with frozen T_subcostal-LA labels (candidate set for ranking).

The dense bank (target/subcostal_bank.py) aims every beam at the GB centroid (+ 15 mm jitter): fine to describe T and
to train on, but a ranking evaluated on it is GB-informed (the candidate set itself encodes the GB position).
Here the candidates use only GB-blind information: skin contacts drawn uniformly from the body-surface voxels of the
approach region (as subcostal_bank.region_points), beam towards a uniformly drawn LIVER voxel (+ the same
jitter), the same beam / normal / depth / body-entry rules, roll uniform.
Labels (GB loaded only for labelling): the frozen T_LA criteria of target/definition.py with subcostal_bank's
relative-area reference (largest GB cut over all approaches), so in_T is exactly T_subcostal-LA.
Output: work/target_bank_blind/<study>.npz (as subcostal_bank), work/target_bank_blind_summary.csv
Usage: python -m sonobelief.target.blind_bank --split train [--shard i/n] [--studies ...]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from sonobelief import fan as G
from sonobelief import paths
from sonobelief.table import load_table
from sonobelief.target.approach import landmarks, margin_profile
from sonobelief.target.definition import CONFIG, label, load_config
from sonobelief.target.ruq_bank import AIM_JITTER_MM, LAB_LIVER, MAX_NORMAL_DEG, Patient, fan_grid, score
from sonobelief.target.subcostal_bank import region_points

OUT = os.path.join(paths.WORK, "target_bank_blind")
N_BLIND = 8000


def candidates(pt, liver_pts, skin_pts, inward_pts, ap, n, gen):
    rnd = lambda *sz: torch.rand(*sz, device="cuda", generator=gen)
    out, have = {k: [] for k in ("apex", "beam", "lateral", "skin", "target")}, 0
    sin_c, cos_p = math.sin(math.radians(ap["min_cephalad_deg"])), math.cos(math.radians(ap["max_from_posterior_deg"]))
    for _ in range(400):
        m = 4 * n
        i = torch.randint(len(skin_pts), (m,), device="cuda", generator=gen)
        skin = skin_pts[i] + (rnd(m, 3) - 0.5) * pt.sp
        inward = inward_pts[i]
        j = torch.randint(len(liver_pts), (m,), device="cuda", generator=gen)
        target = liver_pts[j] + AIM_JITTER_MM * torch.randn(m, 3, device="cuda", generator=gen)
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
    return {k: torch.cat(v)[:n] for k, v in out.items()}, min(have, n)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--split", default="train")
    ap_.add_argument("--studies", nargs="+")
    ap_.add_argument("--shard", default="0/1")
    args = ap_.parse_args()
    cfg = load_config(CONFIG)
    ap = cfg["approach"]
    os.makedirs(OUT, exist_ok=True)
    table = load_table(args.split)
    sids = args.studies or list(table.index)
    si, sn = map(int, args.shard.split("/"))
    sids = sids[si::sn]
    X, Y, in_fan, edge, polar = fan_grid()
    rows, t0 = [], time.time()
    for k, sid in enumerate(sids):
        if os.path.exists(os.path.join(OUT, f"{sid}.csv")):
            continue
        row = table.loc[sid]
        pt = Patient(row)
        costal, sp, mid_x, spine_y = landmarks(row)
        edges, zcm = margin_profile(costal, sp, mid_x, spine_y, ap)
        skin_pts, inward_pts = region_points(pt, mid_x, edges, zcm, ap)
        liver_pts = torch.from_numpy(np.argwhere(pt.lab_np == LAB_LIVER)).cuda().float() * pt.sp
        g = torch.Generator(device="cuda").manual_seed(int(sid[1:]) + 11)
        n_got = 0
        if len(skin_pts) and len(liver_pts):
            P, n_got = candidates(pt, liver_pts, skin_pts, inward_pts, ap, N_BLIND, g)
        r = {"study": sid, "split": args.split, "n_region_skin": len(skin_pts), "n_candidates": n_got}
        if n_got == 0:
            pd.DataFrame([{**r, "n_T_sub": 0}]).to_csv(os.path.join(OUT, f"{sid}.csv"), index=False)
            print(f"[{k + 1}/{len(sids)}] {sid}: no feasible candidates", flush=True)
            continue
        # ---- labels (GB used only from here on)
        s = {key: v.cpu().numpy() for key, v in score(pt, P, X, Y, in_fan, edge, polar).items()}
        ref = np.load(os.path.join(paths.WORK, "gb_target_subcostal", f"{sid}.npz"))
        area_max = float(ref["area_max"])                       # the same relative-area reference as T_subcostal-LA
        comp = label(s, cfg, area_max)
        n_T = int(comp["in_T"].sum())
        np.savez(os.path.join(OUT, f"{sid}.npz"), **{key: v.cpu().numpy() for key, v in P.items()}, **s,
                 **{key: np.asarray(v) for key, v in comp.items()}, area_max=area_max, target_config=json.dumps(cfg))
        r["n_T_sub"] = n_T
        pd.DataFrame([r]).to_csv(os.path.join(OUT, f"{sid}.csv"), index=False)
        msg = f"[{k + 1}/{len(sids)}] {sid}: candidates {n_got}"
        if args.split != "test":
            msg += f" | T_sub {n_T}"
        print(msg + f" | {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
