---
license: mit
library_name: pytorch
tags:
  - medical-imaging
  - ultrasound
  - simulation
  - flow-matching
---

# SonoBelief weights

Frozen components of SonoBelief. Code: [github.com/daryataratynova/sonobelief](https://github.com/daryataratynova/sonobelief).

Trained on TotalSegmentator v2.0.1 (302 train studies), selected on validation (38 studies), seed 0.
The models take simulated label-map views rendered from CT, not ultrasound images.

| Folder | Component |
|---|---|
| `image_encoder/` | E_img: view → 128-d embedding |
| `liver_shape/` | E_L: liver mask → shape code z_L (32-d) |
| `observation_model/` | D_obs: (z_L, probe pose) → view embedding, plus σ² |
| `flow_planes/` | placement flow pre-trained on random views |
| `flow/` | placement flow fine-tuned on sweeps (used) |
| `pose_head/` | g_pose: target score from pose only |
| `geometry_head/` | g_geom: target score from pose + inferred GB geometry (SonoBelief) |
| `exact_geometry_head/` | g_GB: target score from pose + true GB geometry (analysis only) |

```bash
python scripts/weights.py download
```

Not a medical device.
