"""Select the SonoBelief dataset from TotalSegmentator v2.01.

Inclusion (all required):
  - liver mask non-empty and not truncated by the field of view
  - gallbladder mask non-empty and not truncated by the field of view
  - full abdominal coverage: vertebrae T12, L1, L2, L3, L4, L5 all visible

Input:  liver_gb_coverage.csv (from prep/coverage.py)
Output: dataset.csv (one row per included study)
"""
import os

import pandas as pd

from sonobelief import paths

DATA_OUT = paths.WORK
INTER = os.path.join(DATA_OUT, "intermediate")
DATA = paths.TOTALSEG


def main():
    cov = pd.read_csv(os.path.join(INTER, "liver_gb_coverage.csv"), index_col="study")
    meta = pd.read_csv(os.path.join(DATA, "meta.csv"), sep=";", encoding="utf-8-sig").set_index("image_id")

    keep = cov.liver_complete & cov.gallbladder_complete & cov.abd_full
    ds = cov.loc[keep, ["study_type", "liver_ml", "gallbladder_ml"]].join(
        meta[["split", "age", "gender", "institute", "manufacturer", "pathology", "pathology_location"]]
    )
    ds["ct"] = [os.path.join(DATA, s, "ct.nii.gz") for s in ds.index]
    ds["seg_dir"] = [os.path.join(DATA, s, "segmentations") for s in ds.index]
    ds.round(2).to_csv(os.path.join(INTER, "dataset.csv"))

    print(f"{keep.sum()} / {len(cov)} studies included -> dataset.csv")
    print(ds.split.value_counts().to_string())


if __name__ == "__main__":
    main()
