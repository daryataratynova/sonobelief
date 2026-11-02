#!/usr/bin/env bash
# Stage 2: simulated views, target banks and source trajectories for the TotalSegmentator splits.
# Every generator is frozen (configs/target_la.json, configs/source_traj.json); test data is generated but not looked at.
# Slow steps take --shard i/n to split studies over processes.
set -euo pipefail
cd "$(dirname "$0")/.."

for s in train val test; do
  python -m sonobelief.simulate.anatomy --split $s               # 3D crops around the liver (E_L input)
  python -m sonobelief.simulate.planes --split $s                # 512 random liver views per study (E_img / D_obs)
  python -m sonobelief.target.ruq_bank --split $s --all --no_figures   # RUQ candidate bank, T_LA labels, A_max
  python -m sonobelief.target.subcostal_bank --split $s          # dense subcostal bank -> T_subcostal-LA
done
python -m sonobelief.target.applicability                        # long-axis applicability (axis stability)
python -m sonobelief.target.study_labels                         # applicable / available / difficult (f_min: train/val)
for s in train val test; do
  python -m sonobelief.target.blind_bank --split $s              # GB-blind candidates (8000 per study), frozen labels
  python -m sonobelief.simulate.trajectories --split $s --all    # 8 GB-blind subcostal sweeps per study
done
python -m sonobelief.simulate.trajectories --split train --all --augment 64   # train-only augmentation pool
python -m sonobelief.simulate.trajectories --summarise
python -m sonobelief.target.perturbation --dense                 # retention under probe manoeuvres -> tolerances
