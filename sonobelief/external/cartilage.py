"""Costal cartilages and trachea for the external scans (not labelled in ENHANCE; docs/preregistration.md).

TotalSegmentator 2.18.0, task total (1.5 mm model), --roi_subset costal_cartilages trachea, one scan at a time on the
GPU -> work/external/ts_total/<study>/ -> copied as uint8 masks into the study's seg_dir. Only scans that pass the
other inclusion checks are segmented. Table: cartilage_ml, trachea_ml, cartilage_ok (= costal cartilages non-empty),
included (all checks).
Usage: python -m sonobelief.external.cartilage
"""
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import pandas as pd

from sonobelief import paths

TABLE = os.path.join(paths.WORK, "external", "enhance_table.csv")
TS_OUT = os.path.join(paths.WORK, "external", "ts_total")
STRUCT = {"costal_cartilages": "cartilage_ml", "trachea": "trachea_ml"}
TS = os.path.join(os.path.dirname(sys.executable), "TotalSegmentator")


def segment(sid, ct, seg_dir):
    """-> {cartilage_ml, trachea_ml}; TotalSegmentator output cached in work/external/ts_total/<study>/."""
    out = os.path.join(TS_OUT, sid)
    if not all(os.path.exists(os.path.join(out, f"{s}.nii.gz")) for s in STRUCT):
        subprocess.run([TS, "-i", ct, "-o", out, "-ta", "total", "--roi_subset", *STRUCT, "-q"], check=True)
    vol = {}
    for s, col in STRUCT.items():
        img = nib.load(os.path.join(out, f"{s}.nii.gz"))
        m = np.asanyarray(img.dataobj) > 0
        nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine), os.path.join(seg_dir, f"{s}.nii.gz"))
        vol[col] = m.sum() * float(np.prod(img.header.get_zooms()[:3])) / 1000
    return vol


def main():
    df = pd.read_csv(TABLE, index_col="study", dtype={"enhance_id": str})
    todo = df[df.liver_ok & df.gb_ok & df.abd_ok & df.trunk_ok].index
    for k, sid in enumerate(todo):
        for col, v in segment(sid, df.loc[sid, "ct"], df.loc[sid, "seg_dir"]).items():
            df.loc[sid, col] = v
        print(f"[{k + 1}/{len(todo)}] {sid}: cartilage {df.loc[sid, 'cartilage_ml']:.0f} ml, trachea {df.loc[sid, 'trachea_ml']:.0f} ml",
              flush=True)
    df["cartilage_ok"] = df.cartilage_ml.fillna(0) > 0
    df["included"] = df.liver_ok & df.gb_ok & df.abd_ok & df.trunk_ok & df.cartilage_ok
    df.to_csv(TABLE)
    print(f"cartilage_ok {df.cartilage_ok.sum()} / {len(todo)} segmented -> included {df.included.sum()}")


if __name__ == "__main__":
    main()
