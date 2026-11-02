"""Characterisation of T_GB^LA (not part of its definition): perturb poses in T with the standard probe manoeuvres.

For each study, up to N_POSES poses in T (target/ruq_bank.py + target/definition.py) are perturbed:
  slide_inplane   move the skin contact along the skin, in the direction of the image plane (mm)
  slide_sweep     move the skin contact along the skin, perpendicular to the image plane (mm)
  rock            tilt within the image plane, pivot at the skin contact (deg)
  fan             tilt out of the image plane, pivot at the skin contact (deg)
  rotate          rotate about the beam axis (deg)
at magnitudes MAGS (0 = re-render check), with a random sign per pose. Slides keep the orientation; the new
contact is the outermost body point on the skin normal through the shifted point. Each perturbed pose is
re-rendered (shadows included) and labelled with the same target definition (configs/target_la.json), with the
study's own largest GB cut as the area reference. Retention = share of perturbed poses that stay in T.
Output: work/perturbation_retention.csv (study, manoeuvre, magnitude, retention, failing components),
        work/qc/perturbation_retention.png
Usage: python -m sonobelief.target.perturbation [--n 100] [--split train]
"""
import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sonobelief import fan as G
from sonobelief import paths
from sonobelief.table import load_table
from sonobelief.target.definition import COMPONENTS, label, load_config
from sonobelief.target.ruq_bank import Patient, fan_grid, score
from sonobelief.target.subcostal_bank import in_approach

matplotlib.use("Agg")


N_POSES = 200
MAGS = {"slide_inplane": (0, 5, 10, 20), "slide_sweep": (0, 5, 10, 20), "rock": (0, 5, 10, 20),
        "fan": (0, 5, 10, 20), "rotate": (0, 5, 10, 20, 30)}
UNIT = {"slide_inplane": "mm", "slide_sweep": "mm", "rock": "deg", "fan": "deg", "rotate": "deg"}


def rotate(v, k, ang):
    """Rotate vectors v (n, 3) about unit axes k (n, 3) by ang (n,) rad (Rodrigues)."""
    c, s = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
    return v * c + torch.cross(k, v, dim=-1) * s + k * (k * v).sum(-1, keepdim=True) * (1 - c)


def outward_normal(pt, p):
    grad = torch.stack([pt.sample(pt.smooth, p + 3 * e, "bilinear") - pt.sample(pt.smooth, p - 3 * e, "bilinear")
                        for e in torch.eye(3, device="cuda")], -1)
    return -F.normalize(grad, dim=-1)


def to_skin(pt, p, n_out):
    """Outermost in-body point on the line p + t n_out, t in [-30, 30] mm (NaN if none)."""
    t = torch.arange(30, -30.5, -0.5, device="cuda")
    pts = p[:, None] + t[None, :, None] * n_out[:, None]
    inside = pt.sample(pt.body, pts) > 0.5
    first = torch.argmax(inside.int(), 1)
    out = pts[torch.arange(len(p)), first]
    out[~inside.any(1)] = float("nan")
    return out


def perturb(pt, P, man, mag, sign):
    skin, beam, lat = P["skin"], P["beam"], P["lateral"]
    normal = torch.cross(lat, beam, dim=-1)
    a = torch.deg2rad(torch.full_like(sign, float(mag))) * sign
    if man.startswith("slide"):
        n_out = outward_normal(pt, skin)
        u = lat if man == "slide_inplane" else normal
        u = F.normalize(u - (u * n_out).sum(-1, keepdim=True) * n_out, dim=-1)   # tangent to the skin
        skin = to_skin(pt, skin + mag * sign[:, None] * u, n_out)
    elif man == "rock":     # tilt within the image plane: about the plane normal
        beam, lat = rotate(beam, normal, a), rotate(lat, normal, a)
    elif man == "fan":      # tilt out of the image plane: about the lateral axis
        beam = rotate(beam, lat, a)
    elif man == "rotate":   # about the beam
        lat = rotate(lat, beam, a)
    return {"skin": skin, "beam": beam, "lateral": lat, "apex": skin - G.R_APEX_MM * beam}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--config", default=os.path.join(paths.CONFIGS, "target_la.json"))
    ap.add_argument("--dense", action="store_true", help="T_subcostal-LA (dense subcostal bank); a perturbed pose must "
                                                          "also stay in A_subcostal")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert args.split != "test", "the test set is closed"
    table = load_table(args.split)
    summ = pd.read_csv(os.path.join(paths.WORK, "gb_target_summary.csv")).set_index("study")
    src = "gb_target_subcostal" if args.dense else "gb_target"
    have = [s for s in table.index if os.path.exists(os.path.join(paths.WORK, src, f"{s}.npz"))
            and bool(summ.loc[s, "long_axis_applicable"])]   # applicable long-axis targets only
    sids = sorted(np.random.default_rng(1).choice(have, min(args.n, len(have)), replace=False))
    X, Y, in_fan, edge, polar = fan_grid()
    rows, t0 = [], time.time()
    for k, sid in enumerate(sids):
        d = np.load(os.path.join(paths.WORK, src, f"{sid}.npz"))
        area_max = float(d["area_max"]) if args.dense else float(d["area_geo"].max())
        vi = np.nonzero(label(d, cfg, area_max)["in_T"])[0]
        if len(vi) == 0:
            continue
        rng = np.random.default_rng(int(sid[1:]))
        vi = rng.choice(vi, min(N_POSES, len(vi)), replace=False)
        pt = Patient(table.loc[sid])
        P0 = {key: torch.from_numpy(d[key][vi]).cuda() for key in ("skin", "beam", "lateral", "apex")}
        g = torch.Generator(device="cuda").manual_seed(int(sid[1:]))
        for man, mags in MAGS.items():
            sign = torch.randint(0, 2, (len(vi),), device="cuda", generator=g).float() * 2 - 1
            for mag in mags:
                P = perturb(pt, P0, man, mag, sign)
                ok = torch.isfinite(P["skin"]).all(1)
                Pn = {key: torch.nan_to_num(v) for key, v in P.items()}
                s = {key: v.cpu().numpy() for key, v in score(pt, Pn, X, Y, in_fan, edge, polar).items()}
                comp = label(s, cfg, area_max)
                okn = ok.cpu().numpy()
                valid = comp["in_T"] & okn
                appr = np.ones_like(okn)
                if args.dense:
                    appr = in_approach(pt, Pn["skin"], Pn["beam"], float(d["midline_x"]), d["margin_edges"], d["margin_z"],
                                       cfg["approach"]).cpu().numpy()
                    valid = valid & appr
                rows.append({"study": sid, "manoeuvre": man, "magnitude": mag, "unit": UNIT[man],
                             "retention": float(valid.mean()), "n": len(vi),
                             **{f"fail_{c}": float((~comp[c] & okn).mean()) for c in COMPONENTS},
                             "fail_approach": float((~appr & okn).mean())})
        print(f"[{k + 1}/{len(sids)}] {sid}: " + " ".join(
            f"{m} {[round(r['retention'], 2) for r in rows if r['study'] == sid and r['manoeuvre'] == m]}" for m in MAGS)
            + f" | {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    tag = "_subcostal" if args.dense else ""
    df.to_csv(os.path.join(paths.WORK, f"perturbation_retention{tag}.csv"), index=False)

    agg = df.groupby(["manoeuvre", "magnitude"]).retention.agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    agg.columns = ["median", "q25", "q75"]
    print(agg.round(2).to_string())
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    cols = {"slide_inplane": "#2a78d6", "slide_sweep": "#1baf7a", "rock": "#eb6834", "fan": "#eda100", "rotate": "#6250d6"}
    marks = {"slide_inplane": "o", "slide_sweep": "^", "rock": "s", "fan": "D", "rotate": "v"}
    for ax, mans, xl in ((axs[0], ("slide_inplane", "slide_sweep"), "slide along the skin (mm)"),
                         (axs[1], ("rock", "fan", "rotate"), "tilt / rotation (deg)")):
        for m in mans:
            a = agg.loc[m]
            ax.fill_between(a.index, a.q25, a.q75, color=cols[m], alpha=0.14, lw=0)
            ax.plot(a.index, a["median"], color=cols[m], marker=marks[m], lw=2, label=m.replace("_", " "))
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xl)
        ax.set_ylabel("share of valid poses that stay valid")
        ax.grid(axis="y", color="#e4e4e0")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(("T_subcostal-LA" if args.dense else "T_LA") + f": tolerance to probe manoeuvres ({df.study.nunique()} {args.split} studies, median and IQR)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(paths.QC, f"perturbation_retention{tag}.png"))
    print("done")


if __name__ == "__main__":
    main()
