"""Frozen study labels for the external split (docs/preregistration.md), without touching train/val/test rows.

Same rules as target/applicability.py + target/study_labels.py, with the frozen thresholds of configs/target_la.json:
  applicable  GB long-axis instability (1-voxel erosion, dilation, 20 random boundary perturbations) <= 10 deg
  available   T_subcostal-LA (dense subcostal bank) non-empty
  difficult   available and |T_subcostal-LA| / |sampled A_subcostal poses| < f_min (frozen 0.00525; NOT re-derived)
Writes the label columns into the external rows of work/gb_target_summary.csv (rows added by
target/ruq_bank.py --split external). Prints counts only.
Usage: python -m sonobelief.external.labels [--split external|external2|amos|amos2|amos_all]
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd

from sonobelief import paths
from sonobelief.table import load_table
from sonobelief.target.applicability import axis_instability, pca
from sonobelief.target.definition import CONFIG
from sonobelief.target.locked import locked

TD = os.path.join(paths.REPO, "trajectories")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="external", choices=["external", "external2", "amos", "amos2", "amos_all"])
    split = ap.parse_args().split
    app = json.load(open(CONFIG))["applicability"]
    table = load_table(split)
    path = os.path.join(paths.WORK, "gb_target_summary.csv")
    sub = pd.read_csv(os.path.join(paths.WORK, "gb_target_subcostal_summary.csv")).set_index("study")
    vals = {}                                                        # slow part outside the lock
    for sid in table.index:
        img = nib.as_closest_canonical(nib.load(os.path.join(table.loc[sid, "seg_dir"], "gallbladder.nii.gz")))
        m = np.asanyarray(img.dataobj) > 0
        sp = np.array(img.header.get_zooms()[:3])
        ev, evec = pca(m, sp)
        vals[sid] = {"gb_l1": ev[0], "gb_l2": ev[1], "gb_l3": ev[2], "gb_rho": ev[0] / ev[1],
                     "gb_axis_instability_deg": axis_instability(m, sp, evec[:, 0], np.random.default_rng(int(sid[1:]))),
                     "n_T_sub": sub.loc[sid, "n_T_sub"], "n_candidates_sub": sub.loc[sid, "n_candidates"]}
    with locked(path):
        summ = pd.read_csv(path)
        rows_of = ["amos", "amos2"] if split == "amos_all" else [split]      # the pooled split reuses its cohorts' rows
        ext = summ.split.isin(rows_of) & summ.study.isin(table.index)    # (a smoke-test row outside the cohort is ignored)
        assert set(summ[ext].study) == set(table.index), f"run target/ruq_bank.py --split {split} first"
        for i in summ[ext].index:
            for c, v in vals[summ.loc[i, "study"]].items():
                summ.loc[i, c] = v
        e = summ[ext]
        summ.loc[ext, "applicable"] = e.gb_axis_instability_deg <= app["max_axis_instability_deg"]
        summ.loc[ext, "long_axis_applicable"] = summ.loc[ext, "applicable"]
        summ.loc[ext, "f_T_sub"] = e.n_T_sub / e.n_candidates_sub.where(e.n_candidates_sub > 0)
        summ.loc[ext, "available"] = e.n_T_sub.fillna(0) > 0
        summ.loc[ext, "difficult"] = summ.loc[ext, "available"].astype(bool) & (summ.loc[ext, "f_T_sub"] < app["f_min"])
        summ.to_csv(path, index=False)
    e = summ[ext]
    a, v = e.applicable.astype(bool), e.available.astype(bool)
    print(f"{split}: {len(e)} studies | applicable {a.sum()} | available {v.sum()} | applicable and available "
          f"{(a & v).sum()} | difficult {e.difficult.astype(bool).sum()} (f_min {app['f_min']})")


if __name__ == "__main__":
    main()
