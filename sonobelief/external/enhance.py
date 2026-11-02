"""ENHANCE.PET (130 kept head-to-toe scans) -> the SonoBelief data layout, split "external" (docs/preregistration.md).

Per scan e<id>:
  work/external/seg/e<id>/<structure>.nii.gz   uint8 0/1 masks on the CT grid, TotalSegmentator file names:
      liver (ENHANCE label 8, lesion holes filled; + in-liver tumour mask on contrast scans), gallbladder,
      lung_* lobes, rib_{left,right}_{1..13}, sternum, vertebrae_*, scapula_*, clavicula_*
      (costal_cartilages, trachea: added by external/cartilage.py)
  work/seg_body/e<id>.nii.gz                   TotalSegmentator body as 1 = trunk, 2 = extremities (as scripts/body_seg.sh)
Table: work/external/enhance_table.csv (index study, ab_split = external), with the SonoBelief inclusion checks
(sonobelief/prep/qc.py criteria), trunk_cut_cm from the ENHANCE QC and the inclusion flag. No xiphoid tip
(sternum_present = False): the random planes, used only by the data loader, have no subxiphoid window.
Usage: python -m sonobelief.external.enhance [--ids 0028 ...] [--workers 6]
"""
import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from sonobelief import paths
from sonobelief.prep.qc import ABD_VERT, FOV_TOUCH_MAX, GB_MIN_ML, fov_touch, largest_cc, touches_face, z_range

SRC = paths.ENHANCE
OUT_SEG = os.path.join(paths.WORK, "external", "seg")
SEG_BODY = paths.SEG_BODY
TABLE = os.path.join(paths.WORK, "external", "enhance_table.csv")
ROWS = os.path.join(paths.WORK, "external", "rows")
GT = os.path.join(SRC, "imaging-data", "ground-truth")


def labels():
    txt = open(os.path.join(SRC, "labels.json")).read()
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", txt))


def ts_name(group, name):
    """ENHANCE structure name -> TotalSegmentator file name (None = not needed)."""
    if group == "clin_ct_organs":
        return name if name in ("liver", "gallbladder") or name.startswith("lung_") else None
    if group == "clin_ct_ribs":
        return name
    if group == "clin_ct_vertebrae":
        return name.replace("vertebra_", "vertebrae_") if name.startswith("vertebra_") else None
    if group == "clin_ct_peripheral_bones":
        if name.startswith("scapula_"):
            return name
        if name.startswith("clavicle_"):
            return name.replace("clavicle_", "clavicula_")
    return None


def fill_liver_holes(liv):
    """As ENHANCE qc_liver_vessels.fill_liver_holes: 3D hole filling, then per axial slice."""
    f = ndimage.binary_fill_holes(liv)
    return np.stack([ndimage.binary_fill_holes(f[:, :, z]) for z in range(f.shape[2])], -1)


def one(i, contrast):
    """-> the table row of scan i; cached in work/external/rows/e<id>.json (resumable)."""
    sid = f"e{i}"
    cache = os.path.join(ROWS, f"{sid}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    r = convert(i, sid, contrast)
    json.dump(r, open(cache, "w"))
    return r


def convert(i, sid, contrast):
    out = os.path.join(OUT_SEG, sid)
    os.makedirs(out, exist_ok=True)
    ct_path = os.path.join(SRC, "imaging-data", "images", "CT", f"{i}.nii.gz")
    ct_img = nib.load(ct_path)
    aff, hdr = ct_img.affine, ct_img.header
    save = lambda m, n: nib.save(nib.Nifti1Image(m.astype(np.uint8), aff), os.path.join(out, f"{n}.nii.gz"))
    L = labels()
    folder = {"clin_ct_organs": "Organs", "clin_ct_ribs": "Ribs", "clin_ct_vertebrae": "Vertebrae",
              "clin_ct_peripheral_bones": "Peripheral-Bones"}
    masks = {}
    for g, d in folder.items():
        img = nib.load(os.path.join(GT, d, f"{i}.nii.gz"))
        assert np.allclose(img.affine, aff, atol=1e-3) and img.shape == ct_img.shape, (sid, d)
        lab = np.asanyarray(img.dataobj)
        for k, name in L[g].items():
            n = ts_name(g, name)
            if n is not None:
                masks[n] = lab == int(k)
    tum_p = os.path.join(SRC, "totalseg", "liver_vessels", i, "liver_tumor_in_liver.nii.gz")
    liver = masks["liver"]
    if contrast and os.path.exists(tum_p):
        liver = liver | (np.asanyarray(nib.load(tum_p).dataobj) > 0)
    masks["liver"] = fill_liver_holes(liver)
    for n, m in masks.items():
        save(m, n)
    # body: 1 = trunk, 2 = extremities (scripts/body_seg.sh convention)
    bd = os.path.join(SRC, "totalseg", "body", i)
    tr = np.asanyarray(nib.load(os.path.join(bd, "body_trunc.nii.gz")).dataobj) > 0
    ex = np.asanyarray(nib.load(os.path.join(bd, "body_extremities.nii.gz")).dataobj) > 0
    body = np.where(tr, 1, np.where(ex, 2, 0)).astype(np.uint8)
    nib.save(nib.Nifti1Image(body, aff), os.path.join(SEG_BODY, f"{sid}.nii.gz"))

    # SonoBelief inclusion checks (sonobelief.prep.qc.check, in the canonical RAS orientation)
    can = lambda m: np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(m.astype(np.uint8), aff)).dataobj) > 0
    ct = np.asanyarray(nib.as_closest_canonical(ct_img).dataobj)
    sp = nib.as_closest_canonical(ct_img).header.get_zooms()[:3]
    lv, gb = can(masks["liver"]), can(masks["gallbladder"])
    vert = {v: largest_cc(can(masks[v])) if v in masks else np.zeros_like(lv) for v in ABD_VERT}
    zr = {v: z_range(m) for v, m in vert.items()}
    nz = ct.shape[2]
    vox_ml = float(np.prod(sp)) / 1000
    r = {"study": sid, "enhance_id": i, "ab_split": "external", "ct": ct_path, "seg_dir": out,
         "liver_ml": lv.sum() * vox_ml, "gallbladder_ml": gb.sum() * vox_ml,
         "spacing_mm": "x".join(f"{s:.2f}" for s in sp), "contrast_ok": contrast,
         "liver_face": touches_face(lv), "gb_face": touches_face(gb)}
    r["liver_fov_touch"] = fov_touch(lv, ct) if lv.any() else 0
    r["liver_ok"] = bool(lv.any() and not r["liver_face"] and r["liver_fov_touch"] < FOV_TOUCH_MAX)
    r["gb_ok"] = bool(r["gallbladder_ml"] >= GB_MIN_ML and not r["gb_face"])
    present = all(zr[v] is not None for v in ABD_VERT)
    r["abd_ok"] = bool(present and zr["vertebrae_T12"][1] < nz - 1 and zr["vertebrae_L5"][0] > 0)
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()}


def scan_qc():
    """Per-scan QC of the 130 kept ENHANCE scans: trunk_cut_cm (torso skin cut by the field of view) and contrast_ok,
    from enhance_prep (work/external/enhance_scan_qc.csv) or the shipped copy in data/cohorts/. Descriptive columns
    (sex, age, diagnosis, liver_status) are added when ENHANCE_DIR/fullbody_selection.csv exists."""
    p = os.path.join(paths.WORK, "external", "enhance_scan_qc.csv")
    if not os.path.exists(p):
        p = os.path.join(paths.REPO, "data", "cohorts", "enhance_scan_qc.csv")
    qc = pd.read_csv(p, dtype={"id": str}).set_index("id")
    meta = os.path.join(SRC, "fullbody_selection.csv")
    cols = ["sex", "age", "diagnosis", "liver_status"]
    if os.path.exists(meta):
        qc = qc.join(pd.read_csv(meta, dtype={"id": str}).set_index("id")[cols])
    else:
        qc[cols] = np.nan
    return qc


def keep_ids():
    """The 130 head-to-toe ENHANCE scans kept after visual QC (3 of 133 excluded)."""
    return open(os.path.join(paths.REPO, "data", "cohorts", "enhance_keep_ids.txt")).read().split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT_SEG, exist_ok=True)
    os.makedirs(ROWS, exist_ok=True)
    ids = args.ids or keep_ids()
    sel = scan_qc()
    contrast = {i: bool(sel.contrast_ok.get(i, False)) for i in ids}
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, ids, [contrast[i] for i in ids]))
    df = pd.DataFrame(rows).set_index("study")
    for c in ("sex", "age", "diagnosis", "liver_status", "trunk_cut_cm"):
        df[c] = [sel.loc[i, c] for i in df.enhance_id]
    df["trunk_ok"] = df.trunk_cut_cm == 0
    df["sternum_present"] = False                          # no xiphoid tip: planes without the subxiphoid window
    df["tip_x_mm"] = df["tip_y_mm"] = df["tip_z_mm"] = np.nan
    df["fat_frac"], df["adiposity_tertile"] = np.nan, np.nan
    df["cartilage_ok"] = np.nan                            # set by external/cartilage.py
    df["included"] = df.liver_ok & df.gb_ok & df.abd_ok & df.trunk_ok
    if os.path.exists(TABLE) and args.ids:
        old = pd.read_csv(TABLE, index_col="study", dtype={"enhance_id": str})
        df = pd.concat([old.drop(df.index, errors="ignore"), df]).sort_index()
    df.to_csv(TABLE)
    print(f"{len(df)} scans | liver_ok {df.liver_ok.sum()} gb_ok {df.gb_ok.sum()} abd_ok {df.abd_ok.sum()} "
          f"trunk_ok {df.trunk_ok.sum()} -> included (before cartilage check) {df.included.sum()}")


if __name__ == "__main__":
    main()
