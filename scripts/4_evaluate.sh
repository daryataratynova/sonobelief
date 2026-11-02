#!/usr/bin/env bash
# Stage 4: frozen evaluation on TotalSegmentator val, then the single test evaluation.
# With downloaded weights, first encode each split with the frozen encoders (writes into weights/).
set -euo pipefail
cd "$(dirname "$0")/.."
CFM=weights/flow

for s in train val test; do python -m sonobelief.encode.frozen --split $s; done
python -m sonobelief.encode.trajectories --splits train val test

python -m sonobelief.evaluate.target_belief --split val                       # main metrics (AUPRC, R@k, R_local)
python -m sonobelief.evaluate.anatomy --cfm_run $CFM --split val --K 1 2 4 8 --Mp 4 --n_lap 16 --lm_iters 30 \
  --out atypical_val.json                                                       # val: sigma^2 from val itself
python -m sonobelief.evaluate.atypical --run $CFM --csv atypical_val_per_condition.csv --out atypical_val_summary.json

# test: once, with the frozen configuration (configs/target_belief_frozen.json)
python -m sonobelief.evaluate.target_belief --split test --reopen_test
python -m sonobelief.evaluate.anatomy --cfm_run $CFM --split test --K 1 2 4 8 --Mp 4 --n_lap 16 --lm_iters 30 \
  --sigma_json weights/observation_model/sigma2_trajectories.json --out atypical_test.json
python -m sonobelief.evaluate.atypical --run $CFM --csv atypical_test_per_condition.csv --out atypical_test_summary.json
