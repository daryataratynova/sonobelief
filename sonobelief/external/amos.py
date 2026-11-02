"""AMOS22 external cohort (docs/preregistration.md, Amendment 2): segmentation, inclusion, the first 30 included.

Scans in the pre-registered order (data/cohorts/amos_order.txt; only the downloaded ones). Per scan a<id>:
  TotalSegmentator 2.18.0 `total` (all classes) and `body` -> work/amos/ts/<study>/{total,body}/ (cached)
  work/amos/seg/<study>/<structure>.nii.gz   uint8 masks on the CT grid, TotalSegmentator names:
      liver (AMOS label 6, hole-filled as ENHANCE), gallbladder (AMOS label 4); from TotalSegmentator total: rib_*,
      sternum, vertebrae_*, scapula_*, clavicula_*, lung_* lobes, trachea, costal_cartilages
  work/seg_body/<study>.nii.gz                TotalSegmentator body as 1 = trunk, 2 = extremities
Inclusion (as ENHANCE): liver / gallbladder / T12-L5 checks (sonobelief/prep/qc.py), torso skin not cut by the field of
view (trunk mask within 3 voxels of the in-plane border on no slice; ENHANCE fov_truncation.py), costal cartilages
non-empty. The first 30 included scans are cohort 1, the next 60 cohort 2; stops when cohort 2 is full.
Table work/amos/amos_table.csv (cohort 1 / 2; 0 = examined, not included).
No target label, trajectory or result is looked at.
Usage: python -m sonobelief.external.amos
"""
import json
import os
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd

from sonobelief import paths
from sonobelief.external.enhance import fill_liver_holes
from sonobelief.prep.qc import ABD_VERT, FOV_TOUCH_MAX, GB_MIN_ML, fov_touch, largest_cc, touches_face, z_range

SRC = paths.AMOS
OUT = os.path.join(paths.WORK, "amos")
SEG_BODY = paths.SEG_BODY
TABLE = os.path.join(OUT, "amos_table.csv")
TS = os.path.join(os.path.dirname(sys.executable), "TotalSegmentator")
AMOS_LABEL = {"liver": 6, "gallbladder": 4}
TS_PREFIX = ("rib_", "sternum", "vertebrae_", "scapula_", "clavicula_", "lung_", "trachea", "costal_cartilages")
N_COHORT = (30, 60)          # cohort 1 (Amendment 2), cohort 2 (Amendment 3)


def ct_path(i):
    for d in ("imagesTr", "imagesVa"):
        p = os.path.join(SRC, "amos22", d, f"amos_{i}.nii.gz")
        if os.path.exists(p):
            return p
    return None


def totalseg(ct, out, task, tries=6):
    """TotalSegmentator, retried after a pause (the GPU is shared: a failed run is usually out of GPU memory)."""
    if os.path.exists(os.path.join(out, "DONE")):
        return
    for t in range(tries):
        if subprocess.run([TS, "-i", ct, "-o", out, "-ta", task, "-q"]).returncode == 0:
            open(os.path.join(out, "DONE"), "w").close()
            return
        print(f"  TotalSegmentator {task} failed on {os.path.basename(ct)} (try {t + 1}/{tries}); retrying in 3 min", flush=True)
        time.sleep(180)
    raise RuntimeError(f"TotalSegmentator {task} failed {tries} times on {ct}")


def prepare(i):
    """-> table row of scan i (cached in work/amos/rows/a<id>.json)."""
    sid = f"a{i}"
    cache = os.path.join(OUT, "rows", f"{sid}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    ct = ct_path(i)
    ct_img = nib.load(ct)
    aff = ct_img.affine
    seg = os.path.join(OUT, "seg", sid)
    os.makedirs(seg, exist_ok=True)
    ts = os.path.join(OUT, "ts", sid)
    totalseg(ct, os.path.join(ts, "total"), "total")
    totalseg(ct, os.path.join(ts, "body"), "body")
    save = lambda m, n: nib.save(nib.Nifti1Image(m.astype(np.uint8), aff), os.path.join(seg, f"{n}.nii.gz"))
    lab = np.asanyarray(nib.load(ct.replace("images", "labels")).dataobj)
    masks = {"liver": fill_liver_holes(lab == AMOS_LABEL["liver"]), "gallbladder": lab == AMOS_LABEL["gallbladder"]}
    for f in sorted(os.listdir(os.path.join(ts, "total"))):
        n = f[:-7]
        if f.endswith(".nii.gz") and n.startswith(TS_PREFIX):
            masks[n] = np.asanyarray(nib.load(os.path.join(ts, "total", f)).dataobj) > 0
    for n, m in masks.items():
        save(m, n)
    tr = np.asanyarray(nib.load(os.path.join(ts, "body", "body_trunc.nii.gz")).dataobj) > 0
    ex = np.asanyarray(nib.load(os.path.join(ts, "body", "body_extremities.nii.gz")).dataobj) > 0
    nib.save(nib.Nifti1Image(np.where(tr, 1, np.where(ex, 2, 0)).astype(np.uint8), aff), os.path.join(SEG_BODY, f"{sid}.nii.gz"))
    edge = tr[:3].any((0, 1)) | tr[-3:].any((0, 1)) | tr[:, :3].any((0, 1)) | tr[:, -3:].any((0, 1))
    dz = ct_img.header.get_zooms()[2]

    # inclusion checks (sonobelief.prep.qc.check, canonical RAS)
    can = lambda m: np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(m.astype(np.uint8), aff)).dataobj) > 0
    cimg = nib.as_closest_canonical(ct_img)
    ctv, sp = np.asanyarray(cimg.dataobj), cimg.header.get_zooms()[:3]
    lv, gb = can(masks["liver"]), can(masks["gallbladder"])
    zr = {v: z_range(largest_cc(can(masks[v]))) if v in masks else None for v in ABD_VERT}
    vox_ml = float(np.prod(sp)) / 1000
    r = {"study": sid, "amos_id": i, "ab_split": "amos", "ct": ct, "seg_dir": seg,
         "spacing_mm": "x".join(f"{s:.2f}" for s in sp), "liver_ml": lv.sum() * vox_ml, "gallbladder_ml": gb.sum() * vox_ml,
         "liver_face": touches_face(lv), "gb_face": touches_face(gb)}
    r["liver_fov_touch"] = fov_touch(lv, ctv) if lv.any() else 0
    r["liver_ok"] = bool(lv.any() and not r["liver_face"] and r["liver_fov_touch"] < FOV_TOUCH_MAX)
    r["gb_ok"] = bool(r["gallbladder_ml"] >= GB_MIN_ML and not r["gb_face"])
    present = all(zr[v] is not None for v in ABD_VERT)
    r["abd_ok"] = bool(present and zr["vertebrae_T12"][1] < ctv.shape[2] - 1 and zr["vertebrae_L5"][0] > 0)
    r["trunk_cut_cm"] = round(float(edge.sum() * dz / 10), 1)
    r["trunk_ok"] = r["trunk_cut_cm"] == 0
    r["cartilage_ml"] = float(masks["costal_cartilages"].sum() * np.prod(ct_img.header.get_zooms()[:3]) / 1000) \
        if "costal_cartilages" in masks else 0.0
    r["cartilage_ok"] = r["cartilage_ml"] > 0
    r["included"] = r["liver_ok"] and r["gb_ok"] and r["abd_ok"] and r["trunk_ok"] and r["cartilage_ok"]
    r = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()}
    json.dump(r, open(cache, "w"))
    return r


def main():
    os.makedirs(os.path.join(OUT, "rows"), exist_ok=True)
    order = [i for i in open(os.path.join(paths.REPO, "data", "cohorts", "amos_order.txt")).read().split() if ct_path(i)]
    old = pd.read_csv(TABLE, index_col="study") if os.path.exists(TABLE) else None
    rows, chosen = [], []
    for k, i in enumerate(order):
        r = prepare(i)
        r["order"] = k
        if r["included"]:
            chosen.append(r["study"])
        rows.append(r)
        c = 1 if r["included"] and len(chosen) <= N_COHORT[0] else 2 if r["included"] else 0
        print(f"{r['study']} (order {k}): liver {r['liver_ok']} GB {r['gb_ok']} T12-L5 {r['abd_ok']} torso {r['trunk_ok']} "
              f"cartilage {r['cartilage_ok']} -> {f'cohort {c}' if c else '-'} ({len(chosen)}/{sum(N_COHORT)})", flush=True)
        if len(chosen) >= sum(N_COHORT):
            break
    df = pd.DataFrame(rows).set_index("study")
    df["cohort"] = 0
    df.loc[chosen[:N_COHORT[0]], "cohort"] = 1
    df.loc[chosen[N_COHORT[0]:], "cohort"] = 2
    if old is not None:                                    # cohort 1 is fixed (Amendment 2)
        assert set(old.index[old.cohort == 1]) == set(df.index[df.cohort == 1]), "cohort 1 changed"
    df["sternum_present"] = False                          # no xiphoid tip: planes without the subxiphoid window
    df["tip_x_mm"] = df["tip_y_mm"] = df["tip_z_mm"] = np.nan
    df["fat_frac"], df["adiposity_tertile"] = np.nan, np.nan
    df.to_csv(TABLE)
    n1, n2 = (df.cohort == 1).sum(), (df.cohort == 2).sum()
    print(f"examined {len(df)} of {len(order)} downloaded scans -> cohort 1: {n1} included, cohort 2: {n2} included"
          + ("" if n2 >= N_COHORT[1] else " (cohort 2 incomplete: download the next scans of the order)"))


if __name__ == "__main__":
    main()
