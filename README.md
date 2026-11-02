# SonoBelief

[![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey)](#citation)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-weights-yellow)](https://huggingface.co/daryataratynova8/sonobelief)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

SonoBelief ranks ultrasound probe poses for a long-axis gallbladder view, using only liver views acquired before the
gallbladder is seen. It infers the liver placement and shape from those views, predicts the gallbladder geometry,
and scores candidate poses under that belief.

All data is simulated from CT segmentations.

## Install

```bash
git clone https://github.com/daryataratynova/sonobelief && cd sonobelief
pip install -r requirements.txt && pip install -e .
```

Tested with Python 3.12, PyTorch 2.13, CUDA 13.0.

## Data

Download the datasets yourself and set the paths:

| Dataset | Role | Licence | Variable |
|---|---|---|---|
| [TotalSegmentator v2.0.1](https://zenodo.org/records/10047292) | train / val / test | CC BY 4.0 | `TOTALSEG_DIR` |
| [ENHANCE.PET](https://enhance.pet) | external test | CC BY-NC 4.0 | `ENHANCE_DIR` |
| [AMOS22](https://zenodo.org/records/7262581) | external test | CC BY-SA 4.0 | `AMOS_DIR` |

`data/` has the study IDs, splits, inclusion results and seeded scan orders ([data/README.md](data/README.md)).
No images or image-derived data are included.

## Run with the released weights

```bash
python scripts/weights.py download
scripts/1_prepare_totalseg.sh
scripts/2_simulate.sh
scripts/4_evaluate.sh
```

Expected validation results at K = 8:

| Method | Code name | AUPRC | R@1 | R_local@1 |
|---|---|---|---|---|
| Population prior (K = 0) | `population` | 0.090 | 0.13 | 0.30 |
| Point estimate | `point` | 0.139 | 0.18 | 0.39 |
| Pose posterior | `posterior` | 0.157 | 0.18 | 0.46 |
| SonoBelief | `posterior_geom` | 0.187 | 0.27 | 0.53 |
| Oracle placement (pose-only) | `oracle` | 0.161 | 0.35 | 0.52 |

## Train from scratch

| Script | Stage |
|---|---|
| `scripts/1_prepare_totalseg.sh` | cohort selection (1228 → 378 studies), trunk masks, split |
| `scripts/2_simulate.sh` | simulated views, candidate banks, GB-blind sweeps |
| `scripts/3_train.sh` | E_img, E_L, D_obs, placement flow, pose and geometry heads |
| `scripts/4_evaluate.sh` | validation, then test |
| `scripts/5_external.sh` | ENHANCE.PET and AMOS22 ([pre-registration](docs/preregistration.md)) |

Training takes about 1 h on one RTX PRO 6000. Outputs go to `runs/`, generated data to `work/`.
Models were selected on validation only.

## Citation

Coming soon.

## Licence

Code and weights are MIT. The datasets keep their own licences.
