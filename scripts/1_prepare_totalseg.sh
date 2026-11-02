#!/usr/bin/env bash
# Stage 1: the TotalSegmentator cohort (378 studies, train 302 / val 38 / test 38) -> work/dataset_valid.csv.
# Needs TOTALSEG_DIR (TotalSegmentator v2.0.1) and the TotalSegmentator package for the `body` task (GPU, ~30 min).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TOTALSEG_DIR:?set TOTALSEG_DIR}"

python -m sonobelief.prep.coverage        # liver / GB presence and truncation, all 1228 studies
python -m sonobelief.prep.build_dataset   # 431 candidates
python -m sonobelief.prep.qc              # strict checks -> 378 studies, QC gallery in work/qc/
scripts/body_seg.sh 4                     # trunk masks -> work/seg_body/
python -m sonobelief.prep.adiposity       # L3 adiposity (split stratification)
python -m sonobelief.prep.split           # ab_split: deterministic, seed 0
python -m sonobelief.prep.xiphoid         # xiphoid tip

# the split must match the published one
for s in train val test; do
  diff -q <(sort work/splits/$s.txt) <(sort data/splits/totalseg_$s.txt) >/dev/null \
    || { echo "split $s differs from data/splits/totalseg_$s.txt"; exit 1; }
done
echo "splits match data/splits/"
