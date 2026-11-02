"""3D anatomy crops for E_A: [body, liver, GB] around the liver centroid, per study.

Each study gets a cube of side 2 * HALF_MM centred on the liver centroid, axis-aligned with
the canonical RAS mm frame used by simulate/planes.py (voxel index * spacing), sampled at
VOX_MM with trilinear interpolation of the binary masks (partial volume kept as uint8 0-255).
At training time this cube is resampled again into each P_ref frame (anatomy_data.py).

Output: work/anatomy/<study>.npz
  vol      uint8 (3, N, N, N)  channels body, liver, gallbladder; axes x, y, z
  liver_c  float32 (3,)        liver centroid, mm (the cube centre)
  gb_c     float32 (3,)        gallbladder centroid, mm
  liver_r  float32             max distance liver voxel -> liver centroid, mm
  gb_r     float32             max distance GB voxel -> liver centroid, mm
Usage: python -m sonobelief.simulate.anatomy [--studies ...] [--overwrite]
"""
import argparse
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from sonobelief import paths
from sonobelief.table import load_table

OUT = paths.ANATOMY
SEG_BODY = paths.SEG_BODY
HALF_MM = 160.0
VOX_MM = 2.5
N = int(round(2 * HALF_MM / VOX_MM))  # 128


def load(path):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj) > 0, np.array(img.header.get_zooms()[:3], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*")
    ap.add_argument("--split", help="only this split (e.g. external, which load_table() does not include)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    table = load_table(args.split)
    if args.studies:
        table = table.loc[args.studies]
    for k, (sid, row) in enumerate(table.iterrows()):
        out = os.path.join(OUT, f"{sid}.npz")
        if os.path.exists(out) and not args.overwrite:
            continue
        liver, sp = load(os.path.join(row["seg_dir"], "liver.nii.gz"))
        gb, _ = load(os.path.join(row["seg_dir"], "gallbladder.nii.gz"))
        body, _ = load(os.path.join(SEG_BODY, f"{sid}.nii.gz"))
        lv, gv = np.argwhere(liver) * sp, np.argwhere(gb) * sp
        lc, gc = lv.mean(0), gv.mean(0)
        vol = torch.from_numpy(np.stack([body, liver, gb])).cuda().float()[None]
        g = (torch.arange(N, device="cuda").float() + 0.5) * VOX_MM - HALF_MM
        pts = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1) + torch.from_numpy(lc).float().cuda()
        shape = torch.tensor(liver.shape, device="cuda").float()
        grid = (2 * (pts / torch.from_numpy(sp).cuda()) / (shape - 1) - 1).flip(-1)[None]
        crop = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        np.savez_compressed(out, vol=(crop * 255).round().byte().cpu().numpy(),
                            liver_c=lc.astype(np.float32), gb_c=gc.astype(np.float32),
                            liver_r=np.float32(np.linalg.norm(lv - lc, axis=1).max()),
                            gb_r=np.float32(np.linalg.norm(gv - lc, axis=1).max()))
        if (k + 1) % 25 == 0 or k + 1 == len(table):
            print(f"[{k + 1}/{len(table)}] {sid}", flush=True)


if __name__ == "__main__":
    main()
