"""Clinically plausible approach for the first target family: right-subcostal long-axis GB view.

    T_GB^subcostal-LA = { P : approach(P) and T_GB^LA(P) }      (T_GB^LA frozen, target/definition.py)

Patient-specific landmarks (CT masks, canonical RAS mm: x = patient right, y = anterior, z = head):
  midline        x of the T11-L1 vertebral-body centroid; spine_y its y
  costal margin  z_cm(x): lowest z of the right costal arch (rib_right_7..12 + costal_cartilages) anterior to the
                 spine (y >= spine_y + ANTERIOR_MM), per BIN_MM bin of x, median-smoothed; bins without arch
                 voxels are outside the arch (no subcostal contact there)
approach(P) iff (thresholds in configs/target_la.json "approach"):
  region   skin contact x >= midline - MAX_LEFT_MM;  z_cm(x) - BAND_BELOW_MM <= z <= z_cm(x) + BAND_ABOVE_MM;
           skin-normal azimuth in [AZ_MIN, AZ_MAX] (0 = anterior, +90 = patient right): epigastrium to about the
           anterior axillary line
  beam     cephalad: beam_z >= sin(MIN_CEPHALAD_DEG); posterior: -beam_y >= cos(MAX_FROM_POSTERIOR_DEG)
Output: work/approach/<study>.npz (per candidate: region, beam_ok, approach; costal-margin profile)
Usage: python -m sonobelief.target.approach [--split train] [--studies ...]
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

from sonobelief import paths
from sonobelief.table import load_table
from sonobelief.target.definition import CONFIG

OUT = os.path.join(paths.WORK, "approach")
BIN_MM = 5.0


def landmarks(row):
    seg = lambda n: nib.as_closest_canonical(nib.load(os.path.join(row["seg_dir"], n)))
    ref = seg("liver.nii.gz")
    sp = np.array(ref.header.get_zooms()[:3])
    names = os.listdir(row["seg_dir"])
    load = lambda n: np.asanyarray(seg(n).dataobj) > 0
    costal_n = [n for n in names if n in [f"rib_right_{i}.nii.gz" for i in range(7, 13)] + ["costal_cartilages.nii.gz"]]
    vert_n = [n for n in names if n in ("vertebrae_T11.nii.gz", "vertebrae_T12.nii.gz", "vertebrae_L1.nii.gz")]
    with ThreadPoolExecutor(8) as ex:
        costal = np.logical_or.reduce(list(ex.map(load, costal_n)))
        vert = np.logical_or.reduce(list(ex.map(load, vert_n)))
    v = np.argwhere(vert) * sp
    mid_x, spine_y = float(np.median(v[:, 0])), float(np.median(v[:, 1]))
    return costal, sp, mid_x, spine_y


def margin_profile(costal, sp, mid_x, spine_y, ap):
    c = np.argwhere(costal) * sp
    c = c[(c[:, 1] >= spine_y + ap["anterior_mm"]) & (c[:, 0] >= mid_x - ap["max_left_mm"])]
    edges = np.arange(mid_x - ap["max_left_mm"], c[:, 0].max() + BIN_MM, BIN_MM)
    z = np.full(len(edges) - 1, np.nan)
    for i in range(len(z)):
        m = (c[:, 0] >= edges[i]) & (c[:, 0] < edges[i + 1])
        if m.sum() >= 5:
            z[i] = c[m, 2].min()
    ok = np.isfinite(z)
    zs = z.copy()
    zs[ok] = ndimage.median_filter(z[ok], size=5, mode="nearest")
    return edges, zs


def approach_flags(d, body_smooth, sp, shape, mid_x, edges, zcm, ap):
    skin, beam = torch.from_numpy(d["skin"]).cuda(), torch.from_numpy(d["beam"]).cuda()
    vol = torch.from_numpy(body_smooth).cuda()[None, None]
    spt, sht = torch.tensor(sp, device="cuda", dtype=torch.float32), torch.tensor(shape, device="cuda", dtype=torch.float32)

    def sample(p):
        n = 2 * (p.reshape(1, -1, 1, 1, 3) / spt) / (sht - 1) - 1
        return F.grid_sample(vol, n.flip(-1), mode="bilinear", align_corners=True).reshape(p.shape[:-1])
    grad = torch.stack([sample(skin + 3 * e) - sample(skin - 3 * e) for e in torch.eye(3, device="cuda")], -1)
    n_out = (-F.normalize(grad, dim=-1)).cpu().numpy()
    az = np.degrees(np.arctan2(n_out[:, 0], n_out[:, 1]))
    sk, bm = d["skin"], d["beam"]
    idx = np.clip(((sk[:, 0] - edges[0]) / BIN_MM).astype(int), -1, len(zcm))
    inside = (idx >= 0) & (idx < len(zcm))
    zc = np.where(inside, zcm[np.clip(idx, 0, len(zcm) - 1)], np.nan)
    region = (inside & np.isfinite(zc) & (sk[:, 0] >= mid_x - ap["max_left_mm"])
              & (sk[:, 2] >= zc - ap["band_below_mm"]) & (sk[:, 2] <= zc + ap["band_above_mm"])
              & (az >= ap["normal_azimuth_deg"][0]) & (az <= ap["normal_azimuth_deg"][1]))
    beam_ok = (bm[:, 2] >= math.sin(math.radians(ap["min_cephalad_deg"]))) & \
              (-bm[:, 1] >= math.cos(math.radians(ap["max_from_posterior_deg"])))
    return region, beam_ok, az


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--split", default="train")
    ap_.add_argument("--studies", nargs="+")
    args = ap_.parse_args()
    ap = json.load(open(CONFIG))["approach"]
    os.makedirs(OUT, exist_ok=True)
    table = load_table(args.split)
    sids = args.studies or list(table.index)
    rows, t0 = [], time.time()
    for k, sid in enumerate(sids):
        row = table.loc[sid]
        d = np.load(os.path.join(paths.WORK, "gb_target", f"{sid}.npz"))
        costal, sp, mid_x, spine_y = landmarks(row)
        edges, zcm = margin_profile(costal, sp, mid_x, spine_y, ap)
        body = np.asanyarray(nib.as_closest_canonical(nib.load(os.path.join(paths.WORK, "seg_body", f"{sid}.nii.gz"))).dataobj) > 0
        sm = ndimage.uniform_filter(body.astype(np.float32), 7)
        region, beam_ok, az = approach_flags(d, sm, sp, body.shape, mid_x, edges, zcm, ap)
        appr = region & beam_ok
        inT = d["in_T"].astype(bool)
        np.savez(os.path.join(OUT, f"{sid}.npz"), region=region, beam_ok=beam_ok, approach=appr, normal_azimuth=az,
                 margin_edges=edges, margin_z=zcm, midline_x=mid_x, spine_y=spine_y, in_T_sub=appr & inT)
        rows.append({"study": sid, "split": args.split, "n_region": int(region.sum()), "n_approach": int(appr.sum()),
                     "n_T": int(inT.sum()), "n_T_sub": int((appr & inT).sum()),
                     "gb_z_minus_margin_mm": float(d["gb_centroid"][2] - np.nanmin(zcm))})
        print(f"[{k + 1}/{len(sids)}] {sid}: approach {appr.sum()} | T {inT.sum()} -> T_sub {(appr & inT).sum()} "
              f"| {(time.time() - t0) / 60:.1f} min", flush=True)
    path = os.path.join(paths.WORK, "approach_summary.csv")
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old[~old.study.isin(df.study)], df]).sort_values("study")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
