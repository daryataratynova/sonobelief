#!/usr/bin/env bash
# Stage 5: the pre-registered external cohorts (docs/preregistration.md). Nothing is trained or tuned on them.
#   ENHANCE.PET  splits external (cohort 1, 20 scans) and external2 (cohort 2, 20 scans)
#   AMOS22       splits amos (cohort 1, 30 scans) and amos2 (cohort 2, 60 scans)
# Usage: scripts/5_external.sh prepare            convert, check and select both datasets (counts only)
#        scripts/5_external.sh generate <split>   frozen data generation for one cohort
#        scripts/5_external.sh evaluate <split>   the single evaluation of one cohort
set -euo pipefail
cd "$(dirname "$0")/.."
CFM=weights/flow
SIGMA=weights/observation_model/sigma2_trajectories.json

case "$1" in
  prepare)
    : "${ENHANCE_DIR:?set ENHANCE_DIR}" "${AMOS_DIR:?set AMOS_DIR}"
    python -m sonobelief.external.enhance_prep        # TotalSegmentator body / liver vessels, scan QC
    python -m sonobelief.external.select_enhance      # seeded order (data/cohorts/enhance_order.txt), inclusion, cartilage
    python -m sonobelief.external.amos                # seeded order (data/cohorts/amos_order.txt), segmentation, inclusion
    ;;
  generate)
    s=$2
    python -m sonobelief.simulate.anatomy --split $s
    python -m sonobelief.simulate.planes --split $s
    python -m sonobelief.encode.frozen --split $s
    python -m sonobelief.target.ruq_bank --split $s --all --no_figures
    python -m sonobelief.target.subcostal_bank --split $s
    python -m sonobelief.external.labels --split $s   # frozen label rules and thresholds
    python -m sonobelief.target.blind_bank --split $s
    python -m sonobelief.simulate.trajectories --split $s --all
    python -m sonobelief.encode.trajectories --splits $s
    ;;
  evaluate)
    s=$2
    python -m sonobelief.evaluate.target_belief --split $s --external_once
    python -m sonobelief.evaluate.anatomy --cfm_run $CFM --split $s --sigma_json $SIGMA \
      --K 1 2 4 8 --Mp 4 --n_lap 16 --lm_iters 30 --out "atypical_$s.json"
    python -m sonobelief.evaluate.atypical --run $CFM --csv "atypical_${s}_per_condition.csv" --out "atypical_${s}_summary.json"
    ;;
  *) echo "usage: $0 prepare | generate <split> | evaluate <split>"; exit 1 ;;
esac
