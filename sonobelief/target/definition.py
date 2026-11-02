"""T_GB^LA: the single target neighbourhood = probe poses producing an acceptable long-axis GB view.

T(P) = V(P) and A(P) and C(P) and Q(P)   (thresholds in configs/target_la.json, frozen on train/val):
  V visible    enough GB in the image plane at a usable depth
               area_geo >= max(area_rel_min * A_max, area_min_mm2), A_max = the patient's largest GB cut,
               depth_mm[0] <= depth_mm_geo <= depth_mm[1]
               (at least half of the best obtainable GB section AND a minimally meaningful absolute amount)
  A aligned    the GB long axis lies in the image plane (align_deg <= align_max_deg) AND is not beam-parallel:
               presentation angle phi_deg >= presentation_min_deg, phi = angle between the in-plane long axis and
               the beam folded to [0, 90] (0 = GB along the beam / vertical on screen, 90 = across the image)
  C contained  GB centred and not clipped, most of the long axis in the image:
               center_deg_geo <= center_max_frac_half_sector * half sector, clip_frac_geo <= clip_max,
               long_frac_geo >= long_frac_min
  Q acoustic   the acoustic path is clear enough: shadow_frac (shadowed share of the GB pixels) <= shadow_frac_max
               (config key "acoustic")
V and C use the geometric GB pixels (*_geo); shadowing enters only through Q. The per-pose measurements come
from target/ruq_bank.py; labelling needs no rendering, so changing a threshold is a relabel (--relabel), not a rerun.
Usage: python -m sonobelief.target.definition --relabel [--config configs/target_la.json]
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from sonobelief import fan as G
from sonobelief import paths

CONFIG = os.path.join(paths.CONFIGS, "target_la.json")
COMPONENTS = ("visible", "aligned", "contained", "acoustic")


def load_config(path=CONFIG):
    return json.load(open(path))


def presentation_angle(beam, lateral, a1):
    """phi (deg): angle between the in-plane projection of the GB long axis a1 and the beam, folded to [0, 90]."""
    return np.degrees(np.arctan2(np.abs(lateral @ a1), np.abs(beam @ a1)))


def label(s, cfg, area_max):
    """s: per-pose measurements (numpy arrays or torch tensors, keys as stored by target/ruq_bank.py);
    area_max: the patient's largest geometric GB cut (mm^2). -> {component: bool array, 'in_T': bool array}."""
    v, a, c, ac = cfg["visible"], cfg["aligned"], cfg["contained"], cfg["acoustic"]
    area = s["area_geo"]
    has = area > 0
    out = {"visible": has & (area >= max(v["area_rel_min"] * float(area_max), v["area_min_mm2"]))
                      & (s["depth_mm_geo"] >= v["depth_mm"][0]) & (s["depth_mm_geo"] <= v["depth_mm"][1]),
           "aligned": (s["align_deg"] <= a["align_max_deg"]) & (s["phi_deg"] >= a["presentation_min_deg"]),
           "contained": has & (s["center_deg_geo"] <= c["center_max_frac_half_sector"] * math.degrees(G.HALF))
                        & (s["clip_frac_geo"] <= c["clip_max"]) & (s["long_frac_geo"] >= c["long_frac_min"]),
           "acoustic": has & (s["shadow_frac"] <= ac["shadow_frac_max"])}
    out["in_T"] = out["visible"] & out["aligned"] & out["contained"] & out["acoustic"]
    return out


def relabel(cfg_path):
    """Relabel the RUQ bank (work/gb_target) and the dense subcostal bank (work/gb_target_subcostal)."""
    cfg = load_config(cfg_path)
    for bank, summ_name, n_col in (("gb_target", "gb_target_summary.csv", "n_T"),
                                   ("gb_target_subcostal", "gb_target_subcostal_summary.csv", "n_T_sub")):
        files = sorted(glob.glob(os.path.join(paths.WORK, bank, "*.npz")))
        summ_path = os.path.join(paths.WORK, summ_name)
        summ = pd.read_csv(summ_path).set_index("study")
        for f in files:
            d = dict(np.load(f))
            d["phi_deg"] = presentation_angle(d["beam"], d["lateral"], d["gb_axes"][0])
            area_max = float(d["area_max"]) if "area_max" in d else float(d["area_geo"].max())
            lab = label(d, cfg, area_max)
            d.update({k: np.asarray(v) for k, v in lab.items()}, target_config=json.dumps(cfg))
            np.savez(f, **d)
            sid = os.path.basename(f)[:-4]
            summ.loc[sid, n_col] = int(lab["in_T"].sum())
            for k in COMPONENTS:
                summ.loc[sid, f"pass_{k}_pct"] = float(lab[k].mean() * 100)
        summ = summ.drop(columns=[c for c in summ.columns if c.startswith("pass_") and c[5:-4] not in COMPONENTS])
        summ.reset_index().to_csv(summ_path, index=False)
        print(f"{bank}: relabelled {len(files)} studies; median {n_col} {summ[n_col].median():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--relabel", action="store_true")
    a = ap.parse_args()
    if a.relabel:
        relabel(a.config)
