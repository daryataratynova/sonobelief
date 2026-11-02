"""Data access for training on the SonoBelief dataset.

    from sonobelief.table import load_table, AnatoDataset
    ds = AnatoDataset("train")             # 302 studies
    item = ds[0]                           # dict: ct, liver, gallbladder (torch tensors), meta (dict)

Volumes are reoriented to RAS (axis 0 L->R, 1 P->A, 2 I->S) and returned at native
spacing (1.5 mm isotropic for all of TotalSegmentator v2.01). CT is float32 HU; masks are
uint8. Cropping, resampling, intensity windowing and augmentation are left to the caller
(pass `transform`, a callable item -> item).

`meta` carries the study's row from work/dataset_valid.csv: split, study_type, age, sex,
liver/gallbladder volume, adiposity (waist_circ_cm, fat_frac, adiposity_tertile, ...)
and the xiphoid tip (tip_i/j/k are voxel indices in the *original* ct.nii.gz orientation,
tip_x/y/z_mm are world RAS mm; NaN when sternum_present is False).
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sonobelief import paths

TABLE = os.path.join(paths.WORK, "dataset_valid.csv")
TABLE_EXT = os.path.join(paths.WORK, "external", "enhance_table.csv")   # external test sets (docs/preregistration.md)
TABLE_AMOS = os.path.join(paths.WORK, "amos", "amos_table.csv")
SPLITS = ("train", "val", "test", "external", "external2", "amos", "amos2", "amos_all")


def load_table(split=None):
    """dataset_valid.csv as a DataFrame indexed by study, optionally filtered to one split. Pre-registered external
    sets (separate tables, never part of load_table()): "external" / "external2" = ENHANCE.PET cohort 1 / 2,
    "amos" / "amos2" / "amos_all" = AMOS22 cohort 1 / 2 / both."""
    if split in ("amos", "amos2", "amos_all"):
        df = pd.read_csv(TABLE_AMOS, index_col="study", dtype={"amos_id": str})
        return df[df.included.astype(bool) & df.cohort.isin({"amos": [1], "amos2": [2], "amos_all": [1, 2]}[split])]
    if split in ("external", "external2"):
        df = pd.read_csv(TABLE_EXT, index_col="study", dtype={"enhance_id": str})
        return df[df.included.astype(bool) & (df.cohort == (1 if split == "external" else 2))]
    df = pd.read_csv(TABLE, index_col="study")
    if split is not None:
        assert split in SPLITS, split
        df = df[df.ab_split == split]
    return df


def _load_ras(path, dtype):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj).astype(dtype), img.affine


class AnatoDataset(Dataset):
    def __init__(self, split, organs=("liver", "gallbladder"), transform=None):
        self.table = load_table(split)
        self.ids = list(self.table.index)
        self.organs = organs
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        row = self.table.loc[sid]
        ct, affine = _load_ras(row["ct"], np.float32)
        item = {"study": sid, "ct": torch.from_numpy(ct), "affine": torch.from_numpy(affine)}
        for o in self.organs:
            m, _ = _load_ras(os.path.join(row["seg_dir"], f"{o}.nii.gz"), np.uint8)
            item[o] = torch.from_numpy(m > 0).to(torch.uint8)
        item["meta"] = row.to_dict()
        return self.transform(item) if self.transform else item
