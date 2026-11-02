"""Liver / gallbladder availability and abdominal coverage in TotalSegmentator v2.01.

Per study:
  - liver / gallbladder present (non-empty mask), volume, and whether the mask is
    truncated by the scan's field of view (touches first/last axial slice)
  - abdominal coverage from vertebra masks (T12..L5) and study_type in meta.csv

Output: liver_gb_coverage.csv (one row per study) + printed summary.
"""
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd

from sonobelief import paths

DATA = paths.TOTALSEG
DATA_OUT = paths.WORK
INTER = os.path.join(DATA_OUT, "intermediate")
ABD_VERT = ["vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"]


def mask_info(path):
    img = nib.as_closest_canonical(nib.load(path))  # RAS: axis 2 = superior-inferior
    m = np.asanyarray(img.dataobj) > 0
    if not m.any():
        return dict(present=False, ml=0.0, cut_inf=False, cut_sup=False)
    zs = np.where(m.any(axis=(0, 1)))[0]
    nz = m.shape[2]
    ml = m.sum() * float(np.prod(img.header.get_zooms()[:3])) / 1000
    return dict(present=True, ml=ml, cut_inf=bool(zs[0] == 0), cut_sup=bool(zs[-1] == nz - 1))


def study(d):
    sid = os.path.basename(d)
    row = {"study": sid}
    for organ in ["liver", "gallbladder"] + ABD_VERT:
        info = mask_info(os.path.join(d, "segmentations", f"{organ}.nii.gz"))
        if organ.startswith("vertebrae"):
            row[organ] = info["present"]
        else:
            for k, v in info.items():
                row[f"{organ}_{k}"] = v
    return row


def main():
    dirs = sorted(d for d in glob.glob(os.path.join(DATA, "s*")) if os.path.isdir(d))
    with ProcessPoolExecutor(20) as ex:
        df = pd.DataFrame(ex.map(study, dirs, chunksize=8)).set_index("study")
    meta = pd.read_csv(os.path.join(DATA, "meta.csv"), sep=";", encoding="utf-8-sig").set_index("image_id")
    df["study_type"] = meta.study_type.str.strip().reindex(df.index)

    for o in ["liver", "gallbladder"]:
        df[f"{o}_complete"] = df[f"{o}_present"] & ~df[f"{o}_cut_inf"] & ~df[f"{o}_cut_sup"]
    v = df[ABD_VERT]
    df["n_abd_vert"] = v.sum(axis=1)
    df["abd_full"] = v.all(axis=1)  # T12..L5 all visible
    df["abd_any"] = v.any(axis=1)
    df["meta_abdomen"] = df.study_type.str.contains("abd|polytrauma|whole body", na=False)
    df.to_csv(os.path.join(INTER, "liver_gb_coverage.csv"))

    n = len(df)
    L, G = df.liver_present, df.gallbladder_present
    Lc, Gc = df.liver_complete, df.gallbladder_complete
    print(f"studies: {n}")
    print(f"liver present {L.sum()}  (complete {Lc.sum()}, FOV-truncated {(L & ~Lc).sum()})")
    print(f"gallbladder present {G.sum()}  (complete {Gc.sum()}, FOV-truncated {(G & ~Gc).sum()})")
    print(f"both present {(L & G).sum()}  | both complete {(Lc & Gc).sum()}")
    print(f"liver complete but no gallbladder {(Lc & ~G).sum()}   liver truncated & no gallbladder {(L & ~Lc & ~G).sum()}")
    print(f"neither {(~L & ~G).sum()}")
    print(f"\nabdomen: full T12-L5 {df.abd_full.sum()} | any T12-L5 vertebra {df.abd_any.sum()} | meta study_type mentions abdomen {df.meta_abdomen.sum()}")
    print(f"full abdomen & liver complete {(df.abd_full & Lc).sum()} | & gallbladder present {(df.abd_full & Lc & G).sum()}")
    print("\nn of T12-L5 vertebrae visible:\n", df.n_abd_vert.value_counts().sort_index().to_string())
    print("\nliver complete, no gallbladder, by study_type:\n", df[Lc & ~G].study_type.value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
