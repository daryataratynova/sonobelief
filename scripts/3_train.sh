#!/usr/bin/env bash
# Stage 3: train every component from scratch, in dependency order (train split; model selection on val only).
# Each step writes runs/<name>_<timestamp>/. Copy (or point configs/*.json and configs/hybrid_frozen.json at) the
# new run folders before the next step; with the published weights this whole stage is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m sonobelief.train.image_encoder --config configs/image_encoder.json --export   # E_img (+ embeddings)
python -m sonobelief.train.liver_shape --config configs/liver_shape.json                 # E_L (+ z_L per split)
python -m sonobelief.train.observation_model --zl_run weights/liver_shape                # D_obs, sigma^2
python -m sonobelief.train.flow_planes --config configs/flow_planes.json                 # placement flow, random views
python -m sonobelief.encode.trajectories --splits train val test                          # trajectory embeddings
python -m sonobelief.encode.trajectories --aug
python -m sonobelief.train.flow_trajectories --config configs/flow.json                  # fine-tune on sweeps (frozen flow)
# sigma^2 of the trajectory likelihoods (true placement, val): copy the json to weights/observation_model/
python -m sonobelief.evaluate.anatomy --cfm_run weights/flow --split val --K 4 8 --lm_iters 30 --no_cfm_occl \
  --out sigma2_trajectories.json
python -m sonobelief.train.pose_head --config configs/pose_head.json                     # g_pose (pose only)
python -m sonobelief.train.pose_head --config configs/exact_geometry_head.json           # g_GB (true geometry)
python -m sonobelief.evaluate.geometry_ceiling --head weights/exact_geometry_head
python -m sonobelief.train.geometry_head --axis krr                                       # g_geom (inferred geometry)
