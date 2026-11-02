"""Random liver views (work/planes/<study>.npz), bit-packed on the GPU, for E_img and D_obs training.
"""
import os

import numpy as np
import torch

from sonobelief.observation import MIN_LIVER_COMPONENT_PX, clean_packed
from sonobelief.paths import PLANES
from sonobelief.table import load_table

_BITS = 2 ** torch.arange(7, -1, -1, dtype=torch.uint8)  # np.packbits order: MSB first


def unpack(packed):
    """(..., W/8) uint8 -> (..., W) float32, on the same device."""
    bits = (packed[..., None] & _BITS.to(packed.device)) > 0
    return bits.flatten(-2).float()


class PlaneSet:
    def __init__(self, split, device="cuda", windows=(0, 1, 2), max_per_study=None,
                 min_component_px=MIN_LIVER_COMPONENT_PX):
        """min_component_px: liver components smaller than this are removed (observation.py);
        0 keeps the raw masks."""
        self.studies = list(load_table(split).index)
        packed, sid, poses = [], [], {k: [] for k in ["apex", "beam", "lateral", "window"]}
        for i, s in enumerate(self.studies):
            z = np.load(os.path.join(PLANES, f"{s}.npz"))
            keep = np.flatnonzero(np.isin(z["window"], windows))[:max_per_study]
            packed.append(z["masks"][keep])
            sid.append(np.full(len(keep), i, np.int32))
            for k in poses:
                poses[k].append(z[k][keep])
        packed = np.concatenate(packed)
        if min_component_px:
            packed = clean_packed(packed, min_component_px)
        self.packed = torch.from_numpy(packed).to(device)
        self.study_idx = torch.from_numpy(np.concatenate(sid)).to(device)
        self.poses = {k: np.concatenate(v) for k, v in poses.items()}
        self.device = device

    def __len__(self):
        return len(self.packed)

    def batches(self, batch_size, shuffle=False, generator=None):
        n = len(self)
        order = torch.randperm(n, device=self.device, generator=generator) if shuffle else torch.arange(n, device=self.device)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            yield unpack(self.packed[idx]), idx

    def study_of(self, idx):
        return [self.studies[i] for i in self.study_idx[idx].tolist()]

    def pose(self, idx):
        idx = idx.cpu().numpy() if torch.is_tensor(idx) else idx
        return {k: v[idx] for k, v in self.poses.items()}
