"""ENHANCE.PET preparation before conversion (enhance.py): TotalSegmentator body and liver-vessel masks, the liver
tumour mask clipped to the liver, and the per-scan QC used for inclusion.

For each of the 130 kept scans (data/cohorts/enhance_keep_ids.txt), inside ENHANCE_DIR:
  totalseg/body/<id>/            TotalSegmentator `body` (body_trunc, body_extremities, ...)
  totalseg/liver_vessels/<id>/   TotalSegmentator `liver_vessels` (liver_vessels, liver_tumor)
                                 + liver_tumor_in_liver.nii.gz: the tumour mask clipped to the ENHANCE liver label
                                 (lesion holes filled, closed, expanded 3 voxels; the ENHANCE label excludes lesions)
QC -> work/external/enhance_scan_qc.csv:
  trunk_cut_cm   length (cm, along z) over which the trunk mask reaches within 3 voxels of the in-plane image border
  contrast_ok    median liver HU (outside vessels) >= 60
The shipped data/cohorts/enhance_scan_qc.csv holds the values used in the paper; differences are reported.
Usage: python -m sonobelief.external.enhance_prep [--workers 8] [--skip_totalseg]
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from sonobelief import paths
from sonobelief.external.enhance import keep_ids

SRC = paths.ENHANCE


def fill_liver_holes(liv):
    """Fill cavities enclosed by the liver label (lesions, intrahepatic IVC/portal channels): 3D, then per axial slice."""
    f = ndimage.binary_fill_holes(liv)
    return np.stack([ndimage.binary_fill_holes(f[:, :, z]) for z in range(f.shape[2])], -1)


def liver_envelope(liv):
    """Generous liver region for clipping tumour masks: filled liver, closed (surface notches), expanded 3 voxels."""
    st = ndimage.generate_binary_structure(3, 1)
    f = ndimage.binary_fill_holes(ndimage.binary_closing(fill_liver_holes(liv), st, iterations=5))
    return ndimage.binary_dilation(f, st, iterations=3)


def edge_cm(p):
    img = nib.load(p)
    m = np.asanyarray(img.dataobj) > 0
    dz = img.header.get_zooms()[2]
    e = m[:3].any((0, 1)) | m[-3:].any((0, 1)) | m[:, :3].any((0, 1)) | m[:, -3:].any((0, 1))
    return round(e.sum() * dz / 10, 1)


def totalseg(i):
    ct = os.path.join(SRC, "imaging-data", "images", "CT", f"{i}.nii.gz")
    for task in ("body", "liver_vessels"):
        out = os.path.join(SRC, "totalseg", task, i)
        if not os.path.exists(out):
            subprocess.run(["TotalSegmentator", "-i", ct, "-o", out, "-ta", task], check=True,
                           stdout=subprocess.DEVNULL)


def one(i):
    d = os.path.join(SRC, "totalseg", "liver_vessels", i)
    v = nib.load(os.path.join(d, "liver_vessels.nii.gz"))
    ves = np.asanyarray(v.dataobj) > 0
    tum = np.asanyarray(nib.load(os.path.join(d, "liver_tumor.nii.gz")).dataobj) > 0
    liv = np.asanyarray(nib.load(os.path.join(SRC, "imaging-data", "ground-truth", "Organs", f"{i}.nii.gz")).dataobj) == 8
    clip = tum & liver_envelope(liv)
    nib.save(nib.Nifti1Image(clip.astype(np.uint8), v.affine), os.path.join(d, "liver_tumor_in_liver.nii.gz"))
    ct = np.asanyarray(nib.load(os.path.join(SRC, "imaging-data", "images", "CT", f"{i}.nii.gz")).dataobj)
    return dict(id=i, trunk_cut_cm=edge_cm(os.path.join(SRC, "totalseg", "body", i, "body_trunc.nii.gz")),
                contrast_ok=bool(np.median(ct[liv & ~ves]) >= 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_totalseg", action="store_true", help="the TotalSegmentator outputs exist already")
    args = ap.parse_args()
    ids = keep_ids()
    if not args.skip_totalseg:
        for i in ids:      # one at a time: TotalSegmentator uses the GPU
            totalseg(i)
    with ProcessPoolExecutor(args.workers) as ex:
        qc = pd.DataFrame(list(ex.map(one, ids))).set_index("id")
    os.makedirs(os.path.join(paths.WORK, "external"), exist_ok=True)
    qc.to_csv(os.path.join(paths.WORK, "external", "enhance_scan_qc.csv"))
    ref = pd.read_csv(os.path.join(paths.REPO, "data", "cohorts", "enhance_scan_qc.csv"), dtype={"id": str}).set_index("id")
    diff = (qc.trunk_cut_cm != ref.trunk_cut_cm) | (qc.contrast_ok != ref.contrast_ok)
    print(f"{len(qc)} scans | torso cut: {(qc.trunk_cut_cm > 0).sum()} | contrast: {qc.contrast_ok.sum()} | "
          f"differ from the shipped QC: {list(qc.index[diff])}")


if __name__ == "__main__":
    main()
