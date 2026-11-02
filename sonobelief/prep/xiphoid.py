"""Locate the xiphoid tip (inferior end of the sternum) for every study in dataset_valid.csv.

TotalSegmentator's sternum label often stops at the xiphisternal junction and leaves the
xiphoid process out. So:
  junction  = most inferior slice of the largest connected component of the sternum mask
              (centroid of its voxels on that slice)
  tip       = from the junction, follow bone (median-filtered HU > BONE_HU) slice by slice
              downward, staying within +-WIN_X_MM of the midline and +-WIN_Y_MM of the
              previous slice's position, until no bone is found (max MAX_XIPH_MM).
              xiphoid_len_mm = tip - junction. 0 means no ossified xiphoid was found; a
              cartilaginous xiphoid is soft-tissue density on CT and can't be traced, so the
              tip is then only a lower bound on how far up it is (use the junction).
Positions are given in voxel indices of the original CT and in world (scanner, RAS mm)
coordinates, plus where the tip sits relative to the liver (mm below the liver dome,
and how far the liver extends below it).

Previews: work/qc/xiphoid/<study>.png  (sagittal + axial through the tip)
Output:   xiphoid.csv, and dataset_valid.csv gets XIPH_COLS joined in
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from sonobelief import paths

matplotlib.use("Agg")

DATA_OUT = paths.WORK
INTER = os.path.join(DATA_OUT, "intermediate")
PREV = os.path.join(paths.QC, "xiphoid")
BONE_HU = 180
WIN_X_MM, WIN_Y_MM = 8, 4  # y window is per slice, so the trace can't jump into nearby structures
# contrast-filled bowel/stomach and enhanced organs reach bone HU; never trace into them
EXCLUDE = ["liver", "stomach", "duodenum", "small_bowel", "colon", "heart", "esophagus"]
MAX_XIPH_MM = 80


def largest_cc(m):
    lab, n = ndimage.label(m)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n > 1 else m


def trace_xiphoid(ct, start, sp, excl):
    """Follow bone inferiorly from start=(x, y, z) (canonical voxel coords). Returns tip (x, y, z)."""
    x0, y, z = float(start[0]), float(start[1]), int(start[2])
    wx, wy = int(round(WIN_X_MM / sp[0])), int(round(WIN_Y_MM / sp[1]))
    tip = np.array([x0, y, z], float)
    for _ in range(int(MAX_XIPH_MM / sp[2])):
        z -= 1
        if z < 0:
            break
        xa, ya = int(round(x0)) - wx, int(round(y)) - wy
        win = ndimage.median_filter(ct[max(xa, 0):xa + 2 * wx + 1, max(ya, 0):ya + 2 * wy + 1, z].astype(np.float32), 3)
        bone = (win > BONE_HU) & ~excl[max(xa, 0):xa + 2 * wx + 1, max(ya, 0):ya + 2 * wy + 1, z]
        if not bone.any():
            break
        lab, n = ndimage.label(bone)
        big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        bx, by = np.argwhere(big).mean(0)
        y = max(ya, 0) + by  # follow the bone front-to-back; keep x on the midline
        tip = np.array([max(xa, 0) + bx, y, z], float)
    return tip


def analyze(row):
    sid = row["study"]
    ct_raw = nib.load(row["ct"])
    ct_img = nib.as_closest_canonical(ct_raw)  # RAS: axis0 L->R, axis1 P->A, axis2 I->S
    aff = ct_img.affine
    sp = ct_img.header.get_zooms()[:3]
    st = np.asanyarray(nib.as_closest_canonical(nib.load(os.path.join(row["seg_dir"], "sternum.nii.gz"))).dataobj) > 0
    lv = np.asanyarray(nib.as_closest_canonical(nib.load(os.path.join(row["seg_dir"], "liver.nii.gz"))).dataobj) > 0
    res = {"study": sid, "sternum_present": bool(st.any())}
    if not st.any():
        return res
    st = largest_cc(st)
    nz = st.shape[2]
    zs = np.where(st.any((0, 1)))[0]
    z_j = int(zs[0])
    junc = np.array([*np.argwhere(st[:, :, z_j]).mean(0), z_j])
    ct = np.asanyarray(ct_img.dataobj)
    excl = np.zeros(ct.shape, bool)
    for o in EXCLUDE:
        excl |= np.asanyarray(nib.as_closest_canonical(nib.load(os.path.join(row["seg_dir"], f"{o}.nii.gz"))).dataobj) > 0
    excl = ndimage.binary_dilation(excl, iterations=1)
    # skin electrodes / external objects: stay at least 3 mm inside the trunk (seg_body, label 1)
    trunk = np.asanyarray(nib.as_closest_canonical(nib.load(os.path.join(DATA_OUT, "seg_body", f"{sid}.nii.gz"))).dataobj) == 1
    excl |= ~ndimage.binary_erosion(trunk, iterations=max(1, int(round(3 / sp[0]))))
    tip = trace_xiphoid(ct, junc, sp, excl)
    z_tip = int(tip[2])
    junc_w = (aff @ np.r_[junc, 1])[:3]
    tip_w = (aff @ np.r_[tip, 1])[:3]
    # tip voxel in the original (non-canonical) CT index space
    tip_orig = np.linalg.inv(ct_raw.affine) @ np.r_[tip_w, 1]

    lz = np.where(lv.any((0, 1)))[0]
    res.update({
        "tip_i": tip_orig[0], "tip_j": tip_orig[1], "tip_k": tip_orig[2],
        "tip_x_mm": tip_w[0], "tip_y_mm": tip_w[1], "tip_z_mm": tip_w[2],
        "junction_x_mm": junc_w[0], "junction_y_mm": junc_w[1], "junction_z_mm": junc_w[2],
        "xiphoid_len_mm": (z_j - z_tip) * sp[2],
        "sternum_len_mm": (zs[-1] - zs[0] + 1) * sp[2],
        "sternum_cut_inf": z_tip == 0,
        "sternum_cut_sup": zs[-1] == nz - 1,
        "tip_below_liver_dome_mm": (lz[-1] - z_tip) * sp[2],
        "liver_below_tip_mm": (z_tip - lz[0]) * sp[2],
    })

    x, y, z = np.round(tip).astype(int)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    sag = ct[x, :, :].T  # (z, y): anterior to the right
    axs[0].imshow(np.clip(sag, -200, 400), cmap="gray", origin="lower", aspect=sp[2] / sp[1])
    for m, c in [(st[x], "yellow"), (lv[x], "red")]:
        if m.any():
            axs[0].contour(m.T, levels=[0.5], colors=[c], linewidths=0.8)
    axs[0].plot(junc[1], junc[2], "m+", ms=10, mew=2)
    axs[0].plot(y, z, "c+", ms=14, mew=2)
    axs[0].set_title(f"sagittal x={x} (anterior right) magenta=junction cyan=tip", fontsize=8)
    ax_sl = ct[::-1, :, z].T  # patient right on image left
    axs[1].imshow(np.clip(ax_sl, -200, 400), cmap="gray", origin="lower", aspect=sp[1] / sp[0])
    for m, c in [(st[::-1, :, z], "yellow"), (lv[::-1, :, z], "red")]:
        if m.any():
            axs[1].contour(m.T, levels=[0.5], colors=[c], linewidths=0.8)
    axs[1].plot(st.shape[0] - 1 - x, y, "c+", ms=14, mew=2)
    axs[1].set_title(f"axial z={z} (xiphoid tip slice)", fontsize=8)
    for a in axs:
        a.set_axis_off()
    fig.suptitle(f"{sid}  xiphoid {res['xiphoid_len_mm']:.0f} mm  tip {res['tip_below_liver_dome_mm']:.0f} mm below liver dome, "
                 f"liver extends {res['liver_below_tip_mm']:.0f} mm below it"
                 + ("  STERNUM CUT" if res["sternum_cut_inf"] else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PREV, f"{sid}.png"), dpi=80)
    plt.close(fig)
    return res


def main():
    os.makedirs(PREV, exist_ok=True)
    ds = pd.read_csv(os.path.join(DATA_OUT, "dataset_valid.csv"), index_col="study")
    if len(sys.argv) > 1:
        ds = ds.loc[sys.argv[1:]]
    with ProcessPoolExecutor(20) as ex:
        x = pd.DataFrame(ex.map(analyze, ds.reset_index().to_dict("records"), chunksize=4)).set_index("study")
    if len(sys.argv) == 1:
        x = x.round(2)
        x.to_csv(os.path.join(INTER, "xiphoid.csv"))
        ds = ds.drop(columns=[c for c in XIPH_COLS if c in ds])
        ds.join(x[XIPH_COLS]).to_csv(os.path.join(DATA_OUT, "dataset_valid.csv"))
        y = x[x.sternum_present]
        print(f"ossified xiphoid traced below the label: {(y.xiphoid_len_mm > 0).sum()} / {len(y)}")
    print(f"sternum present {x.sternum_present.sum()} / {len(x)}")
    print(x.drop(columns=[c for c in x if c.startswith(("tip_", "junction_")) and not c.startswith("tip_below")])
          .describe().round(1).to_string())


XIPH_COLS = ["sternum_present", "tip_i", "tip_j", "tip_k", "tip_x_mm", "tip_y_mm", "tip_z_mm",
             "junction_z_mm", "xiphoid_len_mm", "tip_below_liver_dome_mm", "liver_below_tip_mm"]

if __name__ == "__main__":
    main()
