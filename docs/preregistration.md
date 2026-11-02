# External test set: ENHANCE.PET (pre-registration)

> Pre-registration of the external evaluation, kept as written (dates, amendments and deviations unchanged).
> File and script names refer to the research code base (working name AnatoBelief); the release equivalents are
> given in brackets.

**Written 2026-10-26, before any external data were converted, generated or evaluated.** Nothing in this file may be
changed after the external evaluation has started. Deviations are added at the end under *Deviations*, dated, with
the reason.

## Purpose

An independent external test of the frozen AnatoBelief pipeline (`train/configs/target_belief_frozen.json`) on a
different source: whole-body FDG-PET/CT (AutoPET part of ENHANCE.PET 1.6k; melanoma, lymphoma and negative
controls), with organ labels from a different annotation protocol. It also turns post-hoc val analysis 1 (atypical
anatomy, README T4h) into a confirmatory test.

**Nothing is trained, tuned or selected on this set.** Every model, threshold, width, cut point and hyperparameter is
the frozen one. Each analysis below is run once.

## Data

- **Source:** ENHANCE.PET (`ENHANCE_DIR`), the 130 head-to-toe scans kept after visual QC (`data/cohorts/enhance_keep_ids.txt`).
  License CC BY-NC 4.0. Skin meshes and face renderings stay local.
- **Study ids:** `e<id>` (e.g. `e0028`), so they never collide with TotalSegmentator ids `s<id>`.
- **Split name:** `external`. It is never pooled with train/val/test and never enters any train/val statistic
  (f_min, atypicality cut points, standardisation statistics, residual tuples).

### Segmentation sources

| Structure | Source |
|---|---|
| liver | ENHANCE `Organs` label 8, with lesion holes filled (`fill_liver_holes` of the ENHANCE folder, using the in-liver tumour mask only on contrast scans) |
| gallbladder | ENHANCE `Organs` label 5 |
| lung lobes | ENHANCE `Organs` labels 9–13 |
| ribs 1–12, sternum | ENHANCE `Ribs` |
| vertebrae | ENHANCE `Vertebrae` |
| scapulae, clavicles | ENHANCE `Peripheral-Bones` |
| costal cartilages, trachea | TotalSegmentator 2.18.0 `total` (predicted; not labelled in ENHANCE) |
| body | TotalSegmentator `body` (already in the ENHANCE folder) |

The CT is used at its native spacing (mostly about 0.9 × 0.9 × 2.5 mm), reoriented to RAS. All rendering samples in mm.
Gas is CT < −400 HU inside the body, as before.

### Inclusion (fixed before conversion)

A scan is included if all of the following hold:
1. the AnatoBelief inclusion criteria (README *Inclusion criteria*): the liver doesn't touch any volume face and
   touches the outside-body air in fewer than 15 voxels; the gallbladder is ≥ 1 ml and doesn't touch any face;
   T12–L5 present, with the top of T12 and the bottom of L5 not at the scan edge;
2. the torso skin is not cut by the CT field of view (`trunk_cut_cm == 0` in `fullbody_selection.csv`), because skin
   contact drives the trajectories and the candidate banks;
3. the costal cartilages are segmented (needed for A_subcostal).

Then, exactly as for test, the frozen study labels: the primary analysis uses **applicable and available** patients.
Ranking metrics use patients with ≥ 1 positive in the GB-blind bank. Every exclusion is counted and reported.

## What is generated (frozen generators and rules, unchanged)

- anatomy crops, true z_L (frozen E_L) and r_GB; random planes and their E_img embeddings (needed by the data
  loader only; not used by the evaluation);
- the RUQ bank (A_max), the dense subcostal bank, the applicability / availability / difficulty labels (frozen rules
  and frozen f_min = 0.00525);
- the GB-blind bank (8000 per study);
- 8 source trajectories per study (frozen generator, seeds study × 100 + j, j = 0..7) and their embeddings.

During preparation **only counts and QC** are looked at (studies included, label counts, failures). No metric of the
belief or of the target heads is computed before the evaluation run.

## Analyses

### A. Primary: frozen target belief (`target_belief_val.py --split external` [`sonobelief.evaluate.target_belief`])

The same variants, metrics, K = 0, 1, 2, 4, 8 and patient-level paired bootstrap (5000, percentile 95%) as T4g.

- **H1 (primary, replication of the test result):** posterior − point AUPRC > 0. Assessed at each of K = 1, 2, 4, 8;
  supported at a K if the 95% CI excludes 0.
- **H2:** the first view helps: AUPRC at K = 1 (posterior) > K = 0 population (paired bootstrap).
- **Reported, no hypothesis:** posterior − oracle, posterior + geometry − posterior, AUROC, R@1, R@10, top-1
  distance, region size / precision, per K.

### B. Confirmatory: atypical anatomy (`hybrid_traj.py` on external, then `atypical_belief.py` [`sonobelief.evaluate.anatomy`, `sonobelief.evaluate.atypical`])

Exactly the val analysis of T4h: run settings `--K 1 2 4 8 --Mp 4 --n_lap 16 --lm_iters 30`, σ² from the frozen
`sigma_json`, prior r̄_GB and tertile cut points (15.7 / 25.6 mm) from **train**.

- **H3 (primary for B):** with true placement and the occlusion-aware belief (`oracle_occl`), the gain in the atypical
  stratum at K = 8 is > 0 (patient bootstrap CI excludes 0).
- **H4:** at K = 8, gain(atypical) − gain(typical) > 0, and Spearman ρ(d, gain) > 0 across patients.
- **Reported, no hypothesis:** the same for K = 1, 2, 4; realistic placement (`cfm_occl`); the GB centroid in the P_ref
  frame (`cfm_occl` vs `cfm_prior_z`).

Conditions: all external trajectories with ≥ K pre-acquisition views of included patients with a GB (the analysis
needs r_GB, not the target, so it is not restricted to applicable / available patients; the same as val).

### C. Descriptive: local refinement

From the same run as A: per-manoeuvre corrections of the top-1 pose, reach@k (k = 1, 2) with the frozen widths, the T
sampling-density check; posterior vs oracle at k = 2 with its CI. No hypothesis (the val comparison was post hoc).

## Interpretation rules (fixed now)

- Absolute levels are expected to be lower than on test (different scanner protocol, slice thickness, annotation
  protocol, patient population). A lower absolute level is reported as domain shift, not as a failure of H1–H4.
- H1 and H3 are the headline results. A failed hypothesis is reported as failed; no re-analysis with other settings
  replaces it. Additional analyses after unblinding are labelled post hoc.
- Wording rules of the project apply: "pre-target-acquisition observations"; no claim of autonomous acquisition.

## Amendment 1 (2026-10-26, 13:10): two cohorts

Written before any external result exists. At that point no target label was used for selection and no metric had
been computed: only the conversion of 66 scans and the e0028 smoke test had run. Generation for all 130 scans takes
about 10 hours, so the external set is split into two pre-specified cohorts. Each is evaluated once.

- **Order:** all 130 kept scans in a seeded random order (`numpy.random.default_rng(20261026).permutation` of the
  sorted ids), written to `external/cohort_order.txt` [`data/cohorts/enhance_order.txt`] before any scan is examined (`external/select_cohort1.py` [`sonobelief.external.select_enhance`]).
- **Cohort 1 (split `external`):** the first 20 scans in that order that pass every inclusion check. It is evaluated
  first, with the analyses A–C exactly as above.
- **Cohort 2 (split `external2`):** every other included scan. It is generated and evaluated once later with the
  same commands. It is reported separately, and then pooled.
- **Pooled analysis:** the same analyses A–C on cohorts 1 + 2, run once after cohort 2. H1–H4 are assessed on cohort 1
  and on the pooled set. The pooled set is the larger, better-powered one; cohort 1 alone has limited power for the
  small H1 effect (+0.012 to +0.015 AUPRC on test, 36 patients).
- No setting is changed between the cohorts on the basis of the cohort-1 results.

## Amendment 2 (2026-10-26, 14:25): second external dataset, AMOS22

Written before any AMOS scan was opened. Only the archive index was read (folder names and file sizes), plus
`dataset.json` for the label ids: liver 6, gallbladder 4. Like ENHANCE, AMOS is **test-only**: nothing is trained,
tuned or selected on it, and every component, rule and threshold is the frozen one.

- **Source:** AMOS22 (Zenodo record 7262581, CC BY-SA 4.0): abdominal CTs from Longgang District People's Hospital
  and Longgang District Central Hospital, Shenzhen, with manual organ labels. Only labelled CTs are used
  (`imagesTr` / `imagesVa`, id ≤ 500). MRI (id > 500) and the unlabelled `imagesTs` are excluded.
- **Order and download:** all 300 labelled CT ids in a seeded random order
  (`random.Random(20261026).shuffle` of the sorted ids, `datasets/amos22/ct_order.txt` [`data/cohorts/amos_order.txt`]). The first 60 in that order
  were downloaded (`datasets/amos_partial.py --ct_labelled --n 60`).
- **Cohort:** the first **30** scans in that order that pass every inclusion check. If fewer than 30 of the 60 pass,
  the next scans of the same order are downloaded and checked until 30 pass.
- **Split / ids:** split `amos`, study ids `a<id>`.
- **Segmentation sources:**
  - liver and gallbladder: the AMOS manual labels. The liver gets the same hole filling as ENHANCE (3D, then per axial
    slice).
  - everything AMOS does not label: TotalSegmentator 2.18.0 `total` for ribs, sternum, vertebrae, scapulae,
    clavicles, lung lobes, trachea and costal cartilages; TotalSegmentator `body` for the trunk / extremity body map.
    These blockers and landmarks are therefore **predicted**, unlike ENHANCE where most are annotated. This is
    reported as a property of the dataset.
- **Inclusion:**
  - the same AnatoBelief criteria as ENHANCE (liver, gallbladder ≥ 1 ml, T12–L5);
  - torso skin not cut by the field of view: the TotalSegmentator trunk mask doesn't reach within 3 voxels of the
    in-plane image border on any slice (the ENHANCE `fov_truncation.py` rule [`sonobelief.external.enhance_prep`], `trunk_cut_cm == 0`);
  - costal cartilages segmented.
- **Analyses:** A–C with H1–H4 exactly as above, on the AMOS cohort, run once. AMOS is reported as its own external
  set. ENHANCE (both cohorts) and AMOS are not pooled for the hypotheses; a pooled table of all external patients
  may be shown as descriptive only.

## Amendment 3 (2026-10-26, 18:40): AMOS cohort 2

**Written after the AMOS cohort-1 results were seen** (H1, H3 and H4 supported; H2 not supported). The motivation is
sample size: 24 patients in the primary analysis and 6 atypical. This amendment is therefore not blind to
cohort 1. Cohort 2 itself is blind: none of its scans had been examined, and its size and analyses are fixed here.

- **Cohort 2 (split `amos2`):** the next **60** included scans in the same seeded AMOS order
  (`datasets/amos22/ct_order.txt` [`data/cohorts/amos_order.txt`]), after the 30 of cohort 1. Same download script, inclusion checks, frozen pipeline
  and commands. Cohort 1 membership is unchanged.
- **Primary for cohort 2:** H1–H4 exactly as pre-registered, on cohort 2 alone, run once: a replication on
  independent patients.
- **Pooled AMOS (split `amos_all`, cohorts 1 + 2):** the same analyses once, reported as **secondary**, and labelled
  as an expansion decided after cohort-1 results.
- Every result is reported whatever the outcome. The size is not changed again on the basis of cohort-2 results.

## Amendment 4 (2026-10-26, 21:40): ENHANCE cohort 2 size

**Written after the ENHANCE cohort-1 results were seen.** No cohort-2 scan had been examined beyond the order list.
The reason is compute time: whole-body scans take about 15 min of GPU time per patient.

- **ENHANCE cohort 2 (split `external2`):** the next **20** included scans in the seeded order
  (`external/cohort_order.txt` [`data/cohorts/enhance_order.txt`]) after the 20 of cohort 1, instead of "all remaining", so **40 ENHANCE patients in
  total**. Any further included scans are not used. Cohort 1 membership is unchanged.
- **Revision (21:47, same evening):** first written as 40 for cohort 2; reduced to 20 on the user's decision, for
  compute time. The selection had then examined only the inclusion checks of 2 cohort-2 scans; no cohort-2 data had
  been generated and nothing had been evaluated.
- **Analyses:** H1–H4 on cohort 2 alone, run once. Pooled ENHANCE (cohorts 1 + 2) is secondary, as in Amendment 1.
  Every result is reported whatever the outcome, and the size is not changed again.

## Amendment 5 (2026-10-27, 10:45): reporting by dataset

**Written after all four cohorts had been evaluated and their results seen.** This is a reporting change only:
nothing is re-evaluated, and no model, rule, threshold, hypothesis or case-selection criterion changes.

- **Main reporting unit: the dataset.** ENHANCE = cohorts 1 + 2 (40 patients) and AMOS = cohorts 1 + 2 (90 patients).
  Each is the union of the cohorts' patients, using the per-patient outputs of their single evaluations. Every
  metric is a mean over patients, so pooling adds no new computation of the model. Hypothesis CIs use the same
  patient bootstrap: 5000 resamples, `default_rng(0)`. On each single cohort this reproduces the evaluation's own CI
  to ±0.001. The atypical analysis is `atypical_belief.py` run on the concatenated per-condition files
  (`atypical_{enhance,amos}_pooled_*`). Tertile cuts stay fixed from train.
- **Pooling ENHANCE was not pre-registered,** and the pooled AMOS was pre-registered only as secondary (Amendment 3).
  The decision to report pooled datasets as main was made by the user, for presentation, after the per-cohort
  results were known. The per-cohort tests stay the pre-registered unit of testing. They are all reported in a
  supplementary table (`paper/tables/table_hypotheses_by_cohort.tex`), including **AMOS cohort 1 H2: not
  supported**.
- **The separate `amos_all` pipeline run** (Amendment 3) evaluates the same 90 patients again with fresh Monte
  Carlo samples. It is kept as a consistency check of the pooled AMOS numbers and is not the reported source.

## Deviations

Clarifications of implementation (2026-10-26, during data generation, before any evaluation; none changes an
analysis or a hypothesis):
- **Random planes have no subxiphoid window.** No xiphoid tip is computed for the external scans
  (`sternum_present = False`), so the subxiphoid share of the random planes goes to the general window. The random
  planes are used only by the data loader, never by the evaluation.
- **σ² is the frozen one** in analysis B (`hybrid_traj.py --sigma_json`, `oracle_s1.json` [`weights/observation_model/sigma2_trajectories.json`]). The external residual
  variance is reported only, as `sigma2_split`.
- **Smoke test:** the pipeline was run end to end on one included scan (e0028) to check the code. Only the generation
  counts were printed: T sizes and the trajectory count. These are the same deterministic outputs the full run keeps.
- **Shared summary CSVs:** `trajectories/data/gb_target*_summary.csv` [`work/gb_target*_summary.csv`] gain the external rows. The original rows are
  unchanged in value (checked against the backup in `data/external/backup/`); a few integer columns are now stored
  as float because of the NaN padding.
- **ENHANCE cohort 1, crashed first evaluation attempt (2026-10-26, 18:15):** one GB-blind-bank shard was killed by
  the system (lack of RAM), so 6 of 20 studies had no candidate bank (e0082, e0259, e0432, e0907, e0963, e1005).
  The evaluation stopped with an error at K = 0 on the first of them, before any metric was computed or printed. The
  run folder held only `config.json`; it was renamed `CRASHED_target_belief_external_20261026_181532`. The 6 banks
  were generated with the frozen generator (same seeds), completeness was checked, and the evaluation was then run.
  That run is the single evaluation of cohort 1. `generate_all.sh` now regenerates missing per-study files and
  refuses to mark a cohort done while any file is missing.
- **AMOS cohort 2, no feasible start (2026-10-27):** for 2 of the 60 patients (a0099, a0132) the frozen GB-blind
  trajectory generator found no feasible start pose, so they have no trajectories and contribute no conditions. This
  is the frozen rule, applied as for every other cohort; it is not an exclusion decision. The completeness check of
  `generate_all.sh` had wrongly treated it as missing data and stopped before the evaluation; it now accepts a recorded
  "no feasible start", and the chain was resumed.
