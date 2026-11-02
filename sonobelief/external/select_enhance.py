"""Pre-registered ENHANCE cohorts: the first 20 included scans of a seeded random order (cohort 1), the next 20 (cohort 2).

The order (all 130 kept ENHANCE scans, numpy default_rng(SEED).permutation of the sorted ids) is written to
data/cohorts/enhance_order.txt before any scan is checked. Scans are converted (enhance.one, cached) and checked in
that order; a scan passing the liver / GB / abdomen / torso checks gets its costal cartilages segmented
(totalseg_cartilage.segment); included = all checks incl. cartilage. Selection stops at 20 + 20 included scans.
Table work/external/enhance_table.csv: every examined scan, cohort 1 / 2 (0 = examined, not in a cohort).
Inclusion uses only the segmentation checks: no target label, trajectory or result is looked at.
Usage: python -m sonobelief.external.select_enhance
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from sonobelief import paths
from sonobelief.external.cartilage import segment
from sonobelief.external.enhance import OUT_SEG, ROWS, TABLE, keep_ids, one, scan_qc

SEED, N_COHORT, BATCH = 20261026, (20, 20), 4      # cohort 1 (Amendment 1), cohort 2 (Amendment 4)


def main():
    os.makedirs(OUT_SEG, exist_ok=True)
    os.makedirs(ROWS, exist_ok=True)
    ids = sorted(keep_ids())
    order = [ids[i] for i in np.random.default_rng(SEED).permutation(len(ids))]
    path = os.path.join(paths.REPO, "data", "cohorts", "enhance_order.txt")
    if os.path.exists(path):
        assert open(path).read().split() == order, "cohort order changed"
    else:
        open(path, "w").write("\n".join(order) + "\n")
    sel = scan_qc()
    old = pd.read_csv(TABLE, index_col="study") if os.path.exists(TABLE) else None
    rows, chosen = [], []
    with ProcessPoolExecutor(2) as ex:
        for b in range(0, len(order), BATCH):
            batch = order[b:b + BATCH]
            for i, r in zip(batch, ex.map(one, batch, [bool(sel.contrast_ok.get(i, False)) for i in batch])):
                r.update({c: sel.loc[i, c] for c in ("sex", "age", "diagnosis", "liver_status", "trunk_cut_cm")})
                r["trunk_ok"] = bool(r["trunk_cut_cm"] == 0)
                r["cartilage_ml"] = r["trachea_ml"] = np.nan
                if r["liver_ok"] and r["gb_ok"] and r["abd_ok"] and r["trunk_ok"] and len(chosen) < sum(N_COHORT):
                    r.update(segment(r["study"], r["ct"], r["seg_dir"]))
                r["cartilage_ok"] = bool(r["cartilage_ml"] > 0)
                r["included"] = r["liver_ok"] and r["gb_ok"] and r["abd_ok"] and r["trunk_ok"] and r["cartilage_ok"]
                r["order"] = order.index(i)
                if r["included"] and len(chosen) < sum(N_COHORT):
                    chosen.append(r["study"])
                c = (1 if chosen.index(r["study"]) < N_COHORT[0] else 2) if r["study"] in chosen else 0
                rows.append(r)
                print(f"{r['study']} (order {r['order']}): liver {r['liver_ok']} GB {r['gb_ok']} abd {r['abd_ok']} "
                      f"torso {r['trunk_ok']} cartilage {r['cartilage_ok']} -> {f'cohort {c}' if c else '-'} "
                      f"({len(chosen)}/{sum(N_COHORT)})", flush=True)
            if len(chosen) >= sum(N_COHORT):
                break
    df = pd.DataFrame(rows).set_index("study")
    df["cohort"] = 0
    df.loc[chosen[:N_COHORT[0]], "cohort"] = 1
    df.loc[chosen[N_COHORT[0]:], "cohort"] = 2
    if old is not None:                                    # cohort 1 is fixed (Amendment 1)
        assert set(old.index[old.cohort == 1]) == set(df.index[df.cohort == 1]), "cohort 1 changed"
    df["sternum_present"] = False                          # no xiphoid tip: planes without the subxiphoid window
    df["tip_x_mm"] = df["tip_y_mm"] = df["tip_z_mm"] = np.nan
    df["fat_frac"], df["adiposity_tertile"] = np.nan, np.nan
    df.sort_values("order").to_csv(TABLE)
    print(f"examined {len(df)} scans in the pre-registered order -> cohort 1: {(df.cohort == 1).sum()} included, "
          f"cohort 2: {(df.cohort == 2).sum()} included")


if __name__ == "__main__":
    main()
