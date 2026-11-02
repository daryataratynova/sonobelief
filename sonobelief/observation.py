"""Observation preprocessing applied before every E_img encoding (training AND inference).

clean_observation removes connected liver components smaller than MIN_LIVER_COMPONENT_PX
(~1 cm^2 at fan.PX_MM = 1.85 mm) from channel 0. Such specks are not identifiable
on real ultrasound, and without this filter the encoder maps "a few liver pixels" vs
"no liver" (and small islands appearing beside the main liver) to large latent jumps.
8-connectivity within each frame; frames are never connected to each other.
The FOV channel is left unchanged.
"""
import numpy as np
import torch
from scipy import ndimage

from sonobelief import fan as G

MIN_LIVER_COMPONENT_MM2 = 100.0
MIN_LIVER_COMPONENT_PX = int(round(MIN_LIVER_COMPONENT_MM2 / G.PX_MM**2))  # 29
_STRUCT = np.zeros((3, 3, 3), bool)
_STRUCT[1] = True  # 8-connected in-plane, no connection across frames


def remove_small_liver(liver, min_px=MIN_LIVER_COMPONENT_PX):
    """liver: bool numpy (N, H, W) -> same with components < min_px removed."""
    lab, n = ndimage.label(liver, structure=_STRUCT)
    if n == 0:
        return liver
    size = np.bincount(lab.ravel())
    keep = size >= min_px
    keep[0] = False
    return keep[lab]


def clean_observation(x, min_px=MIN_LIVER_COMPONENT_PX):
    """x: (N, 2, H, W) numpy or torch (any dtype, channels liver, fov) -> cleaned, same type/device."""
    is_t = torch.is_tensor(x)
    arr = x.detach().cpu().numpy() if is_t else x
    liver = remove_small_liver(arr[:, 0] > 0.5, min_px)
    out = arr.copy()
    out[:, 0] = liver.astype(arr.dtype)
    return torch.from_numpy(out).to(x.device) if is_t else out


def clean_packed(packed, min_px=MIN_LIVER_COMPONENT_PX, chunk=8192):
    """Bit-packed (N, 2, H, W/8) uint8 planes -> cleaned, still packed."""
    out = packed.copy()
    for i in range(0, len(packed), chunk):
        liver = np.unpackbits(packed[i:i + chunk, 0], axis=-1).astype(bool)
        out[i:i + chunk, 0] = np.packbits(remove_small_liver(liver, min_px), axis=-1)
    return out
