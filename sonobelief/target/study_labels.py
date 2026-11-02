"""The three frozen study labels for T_subcostal-LA (rules in configs/target_la.json "applicability").

  applicable    the GB long axis is stable: axis instability <= max_axis_instability_deg (target/applicability.py)
  available     at least one valid subcostal long-axis pose exists: |T_subcostal-LA| > 0 in the dense bank
  difficult     available, and the valid neighbourhood is unusually small:
                f = |T_subcostal-LA| / |sampled A_subcostal poses| < f_min = the DIFFICULT_PCT percentile of f over
                applicable train/val studies with a subcostal target
The fraction (not a pose count) keeps the rule independent of how many candidates are sampled. Thresholds come
from train/val only; test studies get the same frozen rules. The labels are not mixed: an inapplicable study keeps
its available / difficult values for separate analysis.
Output: columns applicable, available, difficult, f_T_sub in work/gb_target_summary.csv; f_min in config
Usage: python -m sonobelief.target.study_labels
"""
import json
import os

import numpy as np
import pandas as pd

from sonobelief import paths
from sonobelief.target.definition import CONFIG

DIFFICULT_PCT = 5


def main():
    summ = pd.read_csv(os.path.join(paths.WORK, "gb_target_summary.csv"))
    sub = pd.read_csv(os.path.join(paths.WORK, "gb_target_subcostal_summary.csv"))
    drop = [c for c in ("n_T_sub", "n_candidates_sub", "f_T_sub", "subcostal_available", "difficult_sub", "applicable",
                        "available", "difficult") if c in summ.columns]
    summ = summ.drop(columns=drop).merge(
        sub[["study", "n_T_sub", "n_candidates"]].rename(columns={"n_candidates": "n_candidates_sub"}), on="study", how="left")
    cfg = json.load(open(CONFIG))
    app = cfg["applicability"]
    summ["applicable"] = summ.gb_axis_instability_deg <= app["max_axis_instability_deg"]
    summ["f_T_sub"] = summ.n_T_sub / summ.n_candidates_sub.where(summ.n_candidates_sub > 0)
    summ["available"] = summ.n_T_sub.fillna(0) > 0
    tv = summ[summ.split.isin(["train", "val"]) & summ.applicable & summ.available]
    f_min = float(np.percentile(tv.f_T_sub, DIFFICULT_PCT))
    summ["difficult"] = summ.available & (summ.f_T_sub < f_min)
    summ["long_axis_applicable"] = summ.applicable          # alias used by the analysis scripts
    summ.to_csv(os.path.join(paths.WORK, "gb_target_summary.csv"), index=False)
    for k in ("rule_subcostal_available", "rule_difficult_sub", "T_sub_min", "rule_difficult", "T_min", "rule_long_axis"):
        app.pop(k, None)
    app.update({"rule_applicable": "applicable iff axis instability <= max_axis_instability_deg (half the 20 deg "
                                   "alignment tolerance; 1-voxel erosion, dilation, 20 random boundary perturbations)",
                "rule_available": "available iff T_subcostal-LA (dense subcostal bank) is non-empty",
                "rule_difficult": f"difficult iff available and |T_subcostal-LA| / |sampled A_subcostal poses| < f_min = "
                                  f"{DIFFICULT_PCT}th percentile over applicable, available train/val studies",
                "f_min": round(f_min, 5)})
    json.dump(cfg, open(CONFIG, "w"), indent=1)
    print(f"f_min = {f_min:.4f} (= {f_min * 4000:.0f} of 4000 candidates)")
    for sp, g in summ.groupby("split"):
        print(f"  {sp:5s}: n {len(g)} | applicable {int(g.applicable.sum())} | available {int(g.available.sum())} "
              f"(applicable & available {int((g.applicable & g.available).sum())}) | difficult {int(g.difficult.sum())} "
              f"(applicable & difficult {int((g.applicable & g.difficult).sum())})")


if __name__ == "__main__":
    main()
