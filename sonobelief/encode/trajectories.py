"""Frozen E_img embeddings of the source-trajectory observations (all splits; data generation, no evaluation).

Input: work/source_traj/<study>.npz (frozen generator): obs (n, 3, 128, 128) bit-packed [liver, body/FOV, shadow].
Canonical roll: the frozen planes (work/planes, E_img and D_obs training) have roll theta in [0, pi) about the
beam, i.e. lateral . e2 >= 0 (e2 = fan_geometry.lateral_axis(beam, pi/2)). A trajectory pose with lateral . e2 < 0
shows the same plane mirrored; it is mapped to lateral -> -lateral and image columns reversed (exact, same plane).
Two encodings per frame (frozen E_img = weights/image_encoder, 2 channels [liver, FOV]):
  e_geom  [liver, FOV]                        geometric view, as in E_img / D_obs training (shadow ignored)
  e_occl  [liver & ~shadow, FOV & ~shadow]    shadow as occlusion: shadowed pixels are outside the valid FOV
Output: work/traj_emb/<split>.npz, one row per frame (trajectory order): study, traj, step, length, t_gb,
  apex / beam / lateral (canonical), flipped, shadow_frac (shadowed share of the FOV), e_geom, e_occl (float16).
Training augmentation: --aug embeds work/source_traj_aug (train only, extra seeds of the frozen generator)
  -> work/traj_emb/train_aug.npz (same columns; traj = index within the augmentation file of the study).
Usage: python -m sonobelief.encode.trajectories [--splits train val test] | --aug
"""
import argparse
import os

import numpy as np
import torch

from sonobelief import paths
from sonobelief.geometry import canonical
from sonobelief.image_ae import PlaneAE
from sonobelief.paths import EMB_RUN
from sonobelief.table import load_table

SRC = os.path.join(paths.WORK, "source_traj")
SRC_AUG = os.path.join(paths.WORK, "source_traj_aug")
OUT = os.path.join(paths.WORK, "traj_emb")


def frames(obs, flip):
    """obs (n, 3, H, W) bool, flip (n,) -> geometric and occluded 2-channel inputs (n, 2, H, W) float."""
    obs = torch.where(flip[:, None, None, None], obs.flip(-1), obs)
    liver, fov, sh = obs[:, 0], obs[:, 1], obs[:, 2]
    geom = torch.stack([liver, fov], 1).float()
    occl = torch.stack([liver & ~sh, fov & ~sh], 1).float()
    return geom, occl, (fov & sh).flatten(1).sum(1) / fov.flatten(1).sum(1).clamp(min=1)


@torch.no_grad()
def encode(enc, x, chunk=2048):
    out = []
    for i in range(0, len(x), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(enc(x[i:i + chunk]).float())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--aug", action="store_true", help="train augmentation pool -> train_aug.npz")
    args = ap.parse_args()
    jobs = [("train", SRC_AUG, "train_aug")] if args.aug else [(s, SRC, s) for s in args.splits]
    os.makedirs(OUT, exist_ok=True)
    ae = PlaneAE(2, 128).cuda().eval()
    ae.load_state_dict(torch.load(os.path.join(EMB_RUN, "best.pt"), weights_only=False)["model"])
    for split, src, name in jobs:
        cols = {k: [] for k in ("study", "traj", "step", "length", "t_gb", "apex", "beam", "lateral", "flipped",
                                "shadow_frac", "e_geom", "e_occl")}
        for sid in load_table(split).index:
            path = os.path.join(src, f"{sid}.npz")
            if not os.path.exists(path):
                continue                                   # no feasible start (see source_traj_summary.csv)
            d = np.load(path)
            obs = torch.from_numpy(np.unpackbits(d["obs"], axis=-1).astype(bool)).cuda()
            for j, (n, off) in enumerate(zip(d["length"], d["offset"])):
                beam = torch.from_numpy(d["beam"][j, :n]).float().cuda()
                lat, flip = canonical(beam, torch.from_numpy(d["lateral"][j, :n]).float().cuda())
                geom, occl, shf = frames(obs[off:off + n], flip)
                cols["study"].append(np.full(n, sid)); cols["traj"].append(np.full(n, j)); cols["step"].append(np.arange(n))
                cols["length"].append(np.full(n, n)); cols["t_gb"].append(np.full(n, d["t_gb"][j]))
                cols["apex"].append(d["apex"][j, :n]); cols["beam"].append(d["beam"][j, :n])
                cols["lateral"].append(lat.cpu().numpy()); cols["flipped"].append(flip.cpu().numpy())
                cols["shadow_frac"].append(shf.cpu().numpy())
                cols["e_geom"].append(encode(ae.enc, geom).cpu().numpy().astype(np.float16))
                cols["e_occl"].append(encode(ae.enc, occl).cpu().numpy().astype(np.float16))
        out = {k: np.concatenate(v) for k, v in cols.items()}
        np.savez_compressed(os.path.join(OUT, f"{name}.npz"), **out)
        print(f"{name}: {len(out['step'])} frames, {len(np.unique(out['study']))} studies, "
              f"flipped {out['flipped'].mean():.0%}", flush=True)
