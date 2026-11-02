"""CT-based adiposity proxies at the L3 level for every study in dataset_valid.csv.

No height/weight is available, so BMI cannot be computed. Instead, on the axial slice
through the middle of L3 (the standard body-composition level), we measure:
  waist_circ_cm      perimeter of the body contour (correlates strongly with BMI)
  body_area_cm2      cross-sectional area of the body
  fat_area_cm2       adipose tissue area, HU -190..-30 inside the body
  fat_frac           fat_area / body_area
  ap_cm, lat_cm      anterior-posterior and lateral body diameters
  adiposity_tertile  low / mid / high by fat_frac; cutoffs from studies whose body is fully inside
                     the image at L3
  noise_hu           HU standard deviation inside the (eroded) liver, an image-noise estimate
  noisy              noise_hu > NOISE_MAX: fat_frac is overestimated (noise pushes muscle into the
                     fat HU range even after median filtering); treat its tertile with caution
  l3_truncated       body cut by the image edge at L3: waist/area/diameters left empty, fat_frac
                     is from the visible part only (biased low, lateral subcutaneous fat missing)

The slice is median-filtered (3x3) before fat thresholding; without it, noise in low-dose
scans pushes muscle voxels into the fat HU range.

Body contour = the trunk (body_trunc) from TotalSegmentator's "body" task, run by
run_body_seg.sh into seg_body/. This excludes arms resting against the torso, blankets and
the table, which a plain HU threshold merges into the body.
Previews: work/qc/adiposity/<study>.png (trunk outline + fat).

Output: adiposity.csv, and dataset_valid.csv gets these columns joined in.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

from sonobelief import paths

matplotlib.use("Agg")

DATA_OUT = paths.WORK
INTER = os.path.join(DATA_OUT, "intermediate")
SEG_BODY = os.path.join(DATA_OUT, "seg_body")
PREV = os.path.join(paths.QC, "adiposity")
FAT_HU = (-190, -30)
NOISE_MAX = 35  # HU


def trunk_mask_2d(sid, z):
    seg = nib.as_closest_canonical(nib.load(os.path.join(SEG_BODY, f"{sid}.nii.gz")))
    b = ndimage.binary_fill_holes(np.asanyarray(seg.dataobj[:, :, z]) == 1)  # 1 = body_trunc
    lab, n = ndimage.label(b)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n else b


def analyze(row):
    sid = row["study"]
    ct_img = nib.as_closest_canonical(nib.load(row["ct"]))
    l3 = np.asanyarray(nib.as_closest_canonical(nib.load(os.path.join(row["seg_dir"], "vertebrae_L3.nii.gz"))).dataobj) > 0
    lab, n = ndimage.label(l3)  # drop stray voxels
    if n > 1:
        l3 = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    zs = np.where(l3.any((0, 1)))[0]
    z = int(round((zs[0] + zs[-1]) / 2))  # mid-vertebra slice
    sx, sy = ct_img.header.get_zooms()[:2]
    sl = np.asanyarray(ct_img.dataobj[:, :, z]).astype(np.float32)
    sl = ndimage.median_filter(sl, size=3)

    body = trunk_mask_2d(sid, z)
    fat = body & (sl >= FAT_HU[0]) & (sl <= FAT_HU[1])
    px = sx * sy / 100
    # perimeter of the body contour in mm (contour points are in voxel units -> scale per axis)
    c = max(measure.find_contours(body.astype(np.uint8), 0.5), key=len)
    perim_mm = np.sum(np.hypot(np.diff(c[:, 0]) * sx, np.diff(c[:, 1]) * sy))
    xs, ys = np.where(body)

    nan = float("nan")
    ct = np.asanyarray(ct_img.dataobj)
    lv = np.asanyarray(nib.as_closest_canonical(nib.load(os.path.join(row["seg_dir"], "liver.nii.gz"))).dataobj) > 0
    lv = ndimage.binary_erosion(lv, iterations=3)  # stay away from edges and vessels at the border
    noise = float(ct[lv].std()) if lv.any() else nan

    trunc = bool(body[0].any() or body[-1].any() or body[:, 0].any() or body[:, -1].any())
    nan = float("nan")
    res = {
        "study": sid,
        "l3_z": z,
        "waist_circ_cm": nan if trunc else perim_mm / 10,
        "body_area_cm2": nan if trunc else body.sum() * px,
        "fat_area_cm2": fat.sum() * px,
        "fat_frac": fat.sum() / body.sum(),
        "lat_cm": nan if trunc else (xs.max() - xs.min() + 1) * sx / 10,
        "ap_cm": nan if trunc else (ys.max() - ys.min() + 1) * sy / 10,
        "noise_hu": noise,
        "noisy": noise > NOISE_MAX,
        "l3_truncated": trunc,
    }

    fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
    ax.imshow(np.clip(sl[::-1].T, -200, 300), cmap="gray", origin="lower")
    ov = np.ma.masked_where(~fat[::-1].T, fat[::-1].T)
    ax.imshow(ov, cmap="autumn", alpha=0.45, origin="lower")
    ax.contour(body[::-1].T, levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title(f"{sid}  waist {res['waist_circ_cm']:.0f} cm  fat {100 * res['fat_frac']:.0f}%"
                 + ("  TRUNCATED" if trunc else "") + ("  NOISY" if noise > NOISE_MAX else ""), fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(PREV, f"{sid}.png"))
    plt.close(fig)
    return res


def main():
    os.makedirs(PREV, exist_ok=True)
    path = os.path.join(DATA_OUT, "dataset_valid.csv")
    ds = pd.read_csv(path, index_col="study")
    ds = ds.drop(columns=[c for c in ds.columns if c in ADIP_COLS])
    with ProcessPoolExecutor(20) as ex:
        a = pd.DataFrame(ex.map(analyze, ds.reset_index().to_dict("records"), chunksize=4)).set_index("study")
    cut = a.fat_frac[~a.l3_truncated].quantile([1 / 3, 2 / 3]).tolist()
    a["adiposity_tertile"] = pd.cut(a.fat_frac, [-np.inf, *cut, np.inf], labels=["low", "mid", "high"])
    a = a.round(3)
    a.to_csv(os.path.join(INTER, "adiposity.csv"))
    ds.join(a[[c for c in ADIP_COLS if c in a]]).to_csv(path)

    print(a.describe().round(2).to_string())
    print(f"\ntertile cutoffs (fat_frac): {cut[0]:.3f} / {cut[1]:.3f}")
    print(f"body truncated at L3: {a.l3_truncated.sum()} / {len(a)}")
    g = a.join(ds.gender)
    print("\nby sex (median):\n", g.groupby("gender")[["waist_circ_cm", "fat_frac", "fat_area_cm2"]].median().round(2).to_string())
    print("\ncorr waist vs fat_frac (complete only):", round(a.waist_circ_cm.corr(a.fat_frac), 2))
    print("\nnoise_hu:", a.noise_hu.describe().round(1).to_dict())
    print("noisy:", a.noisy.sum(), list(a.index[a.noisy]))
    print("tertile counts:\n", pd.crosstab(a.adiposity_tertile, a.l3_truncated).to_string())


ADIP_COLS = ["waist_circ_cm", "body_area_cm2", "fat_area_cm2", "fat_frac", "lat_cm", "ap_cm",
             "noise_hu", "noisy", "l3_truncated", "adiposity_tertile",
             "body_touches_edge"]  # last one: stale column from an earlier version, dropped only

if __name__ == "__main__":
    main()
