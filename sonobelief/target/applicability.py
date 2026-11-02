"""Objective study-level flags for the long-axis target (frozen in configs/target_la.json, derived on train/val).

1. Long-axis applicability (is the target well defined?). The GB long axis a1 (PCA of the CT GB mask, mm) is
   recomputed after small segmentation perturbations: 1-voxel erosion, 1-voxel dilation, and N_RANDOM random
   boundary perturbations (drop half of the inner boundary voxels, add half of the outer ones). instability =
   the largest angle between a1 and a perturbed a1. Applicable iff instability <= 10 deg = half of the 20 deg
   alignment tolerance of T (not tuned). rho = l1 / l2 is reported as a descriptor only: on train/val it did not
   separate unstable axes (they spread over rho 1.1-3.0).
2. Difficulty (the target is valid but hard to reach), separate from applicability: difficult iff |T| < T_min,
   T_min = the DIFFICULT_PCT percentile of |T| over applicable train/val studies.
The per-study axis values are computed once (slow, ~15 min) and reused; --recompute forces it.
Output: columns in work/gb_target_summary.csv (gb_l1..3, gb_rho, gb_axis_instability_deg,
        long_axis_applicable, difficult), T_min in configs/target_la.json, work/qc/gb_applicability.png
Usage: python -m sonobelief.target.applicability [--recompute]
"""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from sonobelief import paths
from sonobelief.table import load_table
from sonobelief.target.definition import CONFIG

matplotlib.use("Agg")


N_RANDOM, DIFFICULT_PCT = 20, 5


def pca(m, sp):
    v = np.argwhere(m) * sp
    ev, evec = np.linalg.eigh(np.cov((v - v.mean(0)).T))
    return ev[::-1], evec[:, ::-1]


def axis_instability(m, sp, a1, rng):
    st = ndimage.generate_binary_structure(3, 1)
    er, di = ndimage.binary_erosion(m, st), ndimage.binary_dilation(m, st)
    variants = [er, di]
    inner, outer = np.argwhere(m & ~er), np.argwhere(di & ~m)
    for _ in range(N_RANDOM):
        p = m.copy()
        drop = inner[rng.random(len(inner)) < 0.5]
        add = outer[rng.random(len(outer)) < 0.5]
        p[tuple(drop.T)] = False
        p[tuple(add.T)] = True
        variants.append(p)
    ang = [np.degrees(np.arccos(min(1.0, abs(pca(p, sp)[1][:, 0] @ a1)))) for p in variants if p.sum() > 10]
    return max(ang)


def main():
    recompute = "--recompute" in sys.argv
    summ = pd.read_csv(os.path.join(paths.WORK, "gb_target_summary.csv"))
    cfg = json.load(open(CONFIG))
    app = cfg["applicability"]
    if recompute or "gb_axis_instability_deg" not in summ.columns:
        tables = {sp: load_table(sp) for sp in ("train", "val", "test")}
        rows = []
        for r in summ.itertuples():
            row = tables[r.split].loc[r.study]
            img = nib.as_closest_canonical(nib.load(os.path.join(row["seg_dir"], "gallbladder.nii.gz")))
            m = np.asanyarray(img.dataobj) > 0
            sp = np.array(img.header.get_zooms()[:3])
            ev, evec = pca(m, sp)
            inst = axis_instability(m, sp, evec[:, 0], np.random.default_rng(int(r.study[1:])))
            rows.append({"study": r.study, "gb_l1": ev[0], "gb_l2": ev[1], "gb_l3": ev[2], "gb_rho": ev[0] / ev[1],
                         "gb_axis_instability_deg": inst})
        f = pd.DataFrame(rows)
        summ = summ.drop(columns=[c for c in f.columns if c != "study" and c in summ.columns]).merge(f, on="study")
    max_deg = app["max_axis_instability_deg"]
    summ["long_axis_applicable"] = summ.gb_axis_instability_deg <= max_deg
    tv = summ[summ.split.isin(["train", "val"])]
    t_min = float(np.percentile(tv[tv.long_axis_applicable].n_T, DIFFICULT_PCT))
    summ["difficult"] = summ.n_T < t_min
    summ.to_csv(os.path.join(paths.WORK, "gb_target_summary.csv"), index=False)
    app["T_min"] = t_min
    json.dump(cfg, open(CONFIG, "w"), indent=1)
    for sp, g in summ.groupby("split"):
        print(f"  {sp:5s}: n {len(g)}, not applicable {int((~g.long_axis_applicable).sum())}, difficult "
              f"{int(g.difficult.sum())} (applicable and difficult {int((g.difficult & g.long_axis_applicable).sum())})")
    print(f"T_min = {t_min:.0f}")
    cols = ["study", "split", "gb_rho", "gb_axis_instability_deg", "gb_ml", "gb_length_mm", "n_T"]
    print("not applicable:\n" + summ[~summ.long_axis_applicable][cols].round(2).to_string())
    print("difficult:\n" + summ[summ.difficult][cols].round(2).to_string())
    rho_star = None
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for sp, c, mk in (("train", "#2a78d6", "o"), ("val", "#eb6834", "s")):
        g = summ[summ.split == sp]
        ax.scatter(g.gb_rho, g.gb_axis_instability_deg, s=14, c=c, marker=mk, label=sp, alpha=0.8, lw=0)
    ax.axhline(max_deg, color="#8a8a85", ls="--", lw=1)
    ax.annotate(f"applicable iff instability ≤ {max_deg:g}°", (ax.get_xlim()[1], max_deg), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("ρ = λ1 / λ2 (GB mask PCA)")
    ax.set_ylabel("long-axis instability under mask perturbation (°)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Is the GB long axis well defined? (train / val)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(paths.QC, "gb_applicability.png"))


if __name__ == "__main__":
    main()
