"""Encode one split with the frozen encoders (nothing is trained). Needed once per split after the data is
generated, and for the train/val/test splits when the weights were downloaded instead of trained.

  E_img (weights/image_encoder): every random plane of work/planes/<study>.npz, cleaned as in its training
        -> weights/image_encoder/embeddings_<split>.npz        (same as image_encoder --export)
  E_L   (weights/liver_shape): liver crop of work/anatomy/<study>.npz
        -> weights/liver_shape/z_L_<split>.npz (z_L, study, r_GB, recon_dice)   (same as liver_shape after training)
Usage: python -m sonobelief.encode.frozen --split {train,val,test,external,external2,amos,amos2,amos_all}
"""
import argparse
import json
import os

import numpy as np
import torch

from sonobelief import paths
from sonobelief.datasets.planes import PlaneSet
from sonobelief.image_ae import PlaneAE
from sonobelief.paths import EMB_RUN
from sonobelief.train.liver_shape import LiverSet, ShapeAE, encode_eval


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True,
                    choices=["train", "val", "test", "external", "external2", "amos", "amos2", "amos_all"])
    SPLIT = ap.parse_args().split
    ck = torch.load(os.path.join(EMB_RUN, "best.pt"), weights_only=False)
    ae = PlaneAE(2, ck["config"]["latent_dim"]).cuda().eval()
    ae.load_state_dict(ck["model"])
    planes = PlaneSet(SPLIT, "cuda", min_component_px=ck["config"].get("min_component_px", 0))
    zs = []
    for x, _ in planes.batches(2048):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zs.append(ae.enc(x).float().cpu())
    np.savez_compressed(os.path.join(EMB_RUN, f"embeddings_{SPLIT}.npz"), z=torch.cat(zs).numpy().astype(np.float16),
                        study=np.array(planes.studies)[planes.study_idx.cpu().numpy()], **planes.poses)
    print(f"E_img: {len(planes)} embeddings ({len(planes.studies)} studies)")
    del planes, ae

    zl_run = os.path.join(paths.REPO, json.load(open(os.path.join(paths.CONFIGS, "hybrid_frozen.json")))["zl_run"])
    ck = torch.load(os.path.join(zl_run, "best.pt"), weights_only=False)
    model = ShapeAE(ck["config"]["z_dim"]).cuda()
    model.load_state_dict(ck["model"])
    L = LiverSet(SPLIT)
    z, d, _ = encode_eval(model, L)
    np.savez(os.path.join(zl_run, f"z_L_{SPLIT}.npz"), z_L=z.cpu().numpy(), study=np.array(L.studies),
             r_gb=L.r_gb.cpu().numpy(), recon_dice=d.cpu().numpy())
    print(f"E_L: {len(L)} livers, reconstruction Dice median {d.median().item():.3f} (min {d.min().item():.3f})")


if __name__ == "__main__":
    main()
