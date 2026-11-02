# Data

| File | Content |
|---|---|
| `splits/totalseg_{train,val,test}.txt` | TotalSegmentator study IDs per split (302 / 38 / 38) |
| `cohorts/totalseg.csv` | split, sex, age, adiposity tertile, study labels |
| `cohorts/enhance.csv` | examined ENHANCE.PET scans: inclusion checks, cohort, study labels |
| `cohorts/amos.csv` | the same for AMOS22 |
| `cohorts/enhance_keep_ids.txt` | 130 head-to-toe ENHANCE.PET scans |
| `cohorts/enhance_order.txt` | scan order, `numpy.random.default_rng(20261026)` |
| `cohorts/amos_order.txt` | scan order, `random.Random(20261026)` |
| `cohorts/enhance_scan_qc.csv` | `trunk_cut_cm`, `contrast_ok` per scan |

Cohort 1 / 2: ENHANCE.PET splits `external` / `external2`, AMOS22 splits `amos` / `amos2`; 0 = not included.
Study labels: `applicable`, `available`, `difficult`.
