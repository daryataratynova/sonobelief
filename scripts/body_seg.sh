#!/usr/bin/env bash
# TotalSegmentator "body" task (1 = trunk, 2 = extremities) for every TotalSegmentator study in work/dataset_valid.csv
# -> work/seg_body/<study>.nii.gz. Studies already done are skipped. Usage: scripts/body_seg.sh [parallel_jobs]
set -euo pipefail
: "${TOTALSEG_DIR:?set TOTALSEG_DIR}"
WORK=${SONOBELIEF_WORK:-$(cd "$(dirname "$0")/.." && pwd)/work}
mkdir -p "$WORK/seg_body"
cd "$WORK"
tail -n +2 dataset_valid.csv | cut -d, -f1 | xargs -P "${1:-4}" -I{} bash -c '
  [ -s seg_body/{}.nii.gz ] && exit 0
  TotalSegmentator -i "$TOTALSEG_DIR/{}/ct.nii.gz" -o seg_body/{}.nii.gz -ta body --ml -q >/dev/null 2>seg_body/{}.log \
    && rm -f seg_body/{}.log || echo "FAILED {}"'
echo "done: $(ls seg_body/*.nii.gz | wc -l) masks"
