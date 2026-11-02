# Target definition: T_subcostal-LA (frozen)

**Status:** frozen 2026-10-23. Every threshold was chosen on **train/val only** (visual inspection, sensitivity analysis, dense-sampling QC) and is **not to be adjusted**. The test split was labelled with the frozen rules; only label counts were looked at.

**Machine-readable source of truth:** [`configs/target_la.json`](../configs/target_la.json).
**Implementation:**
- per-pose rules: [`sonobelief/target/definition.py`](../sonobelief/target/definition.py) (`label`);
- approach region: [`sonobelief/target/approach.py`](../sonobelief/target/approach.py), [`sonobelief/target/subcostal_bank.py`](../sonobelief/target/subcostal_bank.py) (`in_approach`);
- study labels: [`sonobelief/target/applicability.py`](../sonobelief/target/applicability.py), [`sonobelief/target/study_labels.py`](../sonobelief/target/study_labels.py).

## 1. What the target is

The target is the **set of probe poses that produce a clinically acceptable longitudinal (long-axis) gallbladder view from the right-subcostal approach**:

T_subcostal-LA = A_subcostal ∩ V ∩ A ∩ C ∩ Q

In words: subcostal ∩ visible ∩ in-plane ∩ not beam-parallel ∩ contained ∩ acoustically acceptable.

- **There is no single target pose.** T is a continuous subset of probe-pose space defined by the rendering rules below; any pose can be labelled by rendering it.
- **The stored candidates are samples of T.** Each study has 4000 candidates sampled densely inside A_subcostal: Monte Carlo samples of that subset.
- **T may be multimodal** (T = T₁ ∪ T₂, e.g. an epigastric and a right-subcostal family). The error of a predicted pose is its distance to the **closest** pose in T.

A pose P = (skin contact, beam direction b, lateral image axis l) of a convex probe: 65° sector, 170 mm imaging depth, virtual apex 50 mm behind the skin ([`sonobelief/fan.py`](../sonobelief/fan.py)). Coordinates are canonical RAS mm (x = patient right, y = anterior, z = head).

## 2. Per-pose criteria

GB geometry is automatic: PCA of the CT GB mask gives the centroid, the long axis a₁ and the length L (2nd–98th percentile extent along a₁).

### V: visible (enough GB, usable depth)

| Measurement | Rule |
|---|---|
| A = GB area cut by the fan (all GB pixels, mm²) | **A ≥ max(0.5 · A_max, 100 mm²)** |
| A_max = the patient's largest GB cut over all candidate poses of all approaches | (reference only) |
| depth of the GB pixel centroid below the skin | **30–150 mm** |

At least half of the best obtainable GB section *and* a minimally meaningful absolute amount (1 cm²).

### A: aligned (long axis in plane and presented as a long-axis view)

| Measurement | Rule |
|---|---|
| θ_plane = angle between a₁ and the image plane | **θ_plane ≤ 20°** |
| φ = presentation angle: in-plane projection of a₁ vs the beam, folded to 0–90° (0° = GB along the beam / vertical on screen, 90° = across the image) | **φ ≥ 45°** |

"The GB long axis lies in the image plane" **and** "the GB runs more across the image than along the beam". It does not require a perfectly horizontal GB.

### C: contained (guard against edge-of-image views)

| Measurement | Rule |
|---|---|
| angle of the GB pixel centroid from the beam axis | **≤ 0.6 × half sector (19.5°)** |
| share of GB pixels within 2 mm of the fan border | **≤ 5%** |
| GB extent along the in-plane long axis / L | **≥ 0.7** |

### Q: acoustic path acceptable

| Measurement | Rule |
|---|---|
| shadowed share of the GB pixels in the fan | **≤ 0.25** |

How the shadow is computed:
- **Rays:** about 200 rays from the skin arc outwards in 1 mm steps; everything beyond the first blocker is shadowed (all-or-nothing).
- **Bone blockers:** ribs, sternum, vertebrae, scapulae, claviculae. Costal cartilage is *not* a blocker.
- **Air blockers:** lung lobes, trachea, and CT voxels < −400 HU inside the body (bowel gas).

V and C are measured on all GB pixels; shadowing enters only through Q.

### A_subcostal: the approach

Our **operationalisation** of the clinically described right-subcostal approach. The numerical bounds are ours, not a clinical standard.

Patient-specific landmarks from the CT masks:
- **midline:** x of the T11–L1 vertebral bodies;
- **costal margin z_cm(x):** per 5 mm of lateral position, the lowest point of the right costal arch (right ribs 7–12 + costal cartilages) anterior to the spine (≥ 30 mm in front of the vertebral body), median-smoothed.

| Part | Rule |
|---|---|
| skin contact, cranio-caudal | **from 10 mm above to 100 mm below the costal margin** at that lateral position |
| skin contact, lateral | **≤ 30 mm left of the midline**, and the outward skin normal **between 30° left of anterior and 60° toward the right** (epigastrium to about the anterior axillary line) |
| beam, posterior | **≥ 30° posterior component** (−b_y ≥ cos 60°) |
| beam, caudal tilt | **≤ 20° toward the feet** (b_z ≥ −sin 20°); cephalad allowed but not required |

## 3. Study-level labels

Three separate labels, never mixed.

| Label | Question | Rule |
|---|---|---|
| **applicable** | Is a long-axis target well defined for this GB? | **Axis instability ≤ 10°**: the largest change of a₁ under small segmentation perturbations (1-voxel erosion, 1-voxel dilation, 20 random boundary perturbations dropping half of the inner and adding half of the outer boundary voxels). 10° = half the 20° in-plane tolerance; not tuned. ρ = λ₁/λ₂ is reported but not used: it did not separate unstable axes. |
| **available** | Does at least one valid subcostal long-axis pose exist? | **T_subcostal-LA ≠ ∅** in the dense subcostal bank. An empty target is a finding, not a failure: that patient would need, e.g., an intercostal window. |
| **difficult** | Is the valid neighbourhood unusually small? | Available **and** f = \|T_subcostal-LA\| / \|sampled A_subcostal poses\| **< f_min = 0.00525**, the 5th percentile of f over applicable, available train/val studies. A fraction, so it does not depend on how many candidates are sampled (≈ 21 of 4000). |

- **Inapplicable studies** stay in the dataset. They are excluded from the primary long-axis experiment, the number excluded is reported, and they are analysed separately.
- **Difficult studies** stay in the main dataset, flagged.

Counts (final definition):

| | Train | Val | Test |
|---|---|---|---|
| applicable | 296 / 302 | 35 / 38 | 38 / 38 |
| available | 300 | 38 | 38 |
| difficult | 17 | 1 | 2 |

Applicable but unavailable (train/val): s0224, s1143.

## 4. Properties of the final target (train/val; characterisation, not part of the definition)

- **Size:** median 166 poses in T per study, of 4000 dense subcostal candidates (IQR 84–240).
- **Usually one neighbourhood:** at a link of about 3× the sampling resolution, one 6D component holds ≥ 90% of T in 80% of studies, and the largest skin patch holds a median 97% (about 12 cm of skin). Some targets have two families (epigastric and right-subcostal).
- **Strongly anisotropic** (share of target poses that stay in T after one manoeuvre, at 5 / 10 / 20 mm or °):
  - in-plane slide 0.89 / 0.81 / 0.65;
  - rock 0.83 / 0.66 / 0.37;
  - rotate 0.81 / 0.65 / 0.39;
  - cross-plane sweep 0.62 / 0.30 / 0.00;
  - fan 0.44 / 0.12 / 0.00.

  Wide along the image plane, narrow across it. This is reported as a property of the actual pose set and **not** converted into a spherical "within x mm and y°" tolerance.

## 5. Pose distance (for errors and connectivity)

d(P, P′) = ‖skin − skin′‖ (mm) + rotation angle between the probe frames (deg)

The rotation angle is taken **modulo the 180° flip about the beam**: roll + 180° shows the same image plane mirrored, and roll is sampled over 0–180°. The error of a prediction P̂ is min over P ∈ T of d(P̂, P).

## 6. Data

| What | Where |
|---|---|
| dense subcostal bank per study: candidate poses, all measurements (incl. `phi_deg`, `shadow_frac`, `*_geo`), component flags, `in_T` = T_subcostal-LA | `work/gb_target_subcostal/<study>.npz` |
| study labels, \|T\|, GB size, adiposity | `work/gb_target_summary.csv` (`applicable`, `available`, `difficult`, `n_T_sub`, `f_T_sub`) |
| RUQ bank (all approaches; T_LA only, used for A_max and analyses) | `work/gb_target/<study>.npz` |

## 7. How the thresholds were chosen (train/val only)

| Rule | Basis |
|---|---|
| θ_plane ≤ 20° | 15° would remove 25% of T without a clinical reason; 20° gives a meaningful longitudinal view without making T artificially thin. |
| A ≥ 0.5 A_max | Interpretable midpoint: at least half of the best obtainable section. |
| A ≥ 100 mm² | A relative-only rule accepted views of under 1 cm² in tiny contracted GBs (s0250: 5 → 261 poses). |
| depth ≤ 150 mm | 120 mm would remove 18% of T for no strong clinical reason; 150 mm still excludes implausibly deep targets. |
| shadow ≤ 0.25 | The shadowed share is bimodal; 0.2–0.3 is a flat, stable region. |
| C (centre, clip, long axis) | Rarely the only failing component; kept as a guard. |
| φ ≥ 45° | Clinical guidance displays the long-axis GB across the screen. Train/val view grids at φ ≈ 15–90° showed 45° as the transition from beam-parallel to oblique/across. Keeps 33% of the previous target; availability unchanged except s0224. |
| A_subcostal bounds | A first version requiring a cephalad beam and a contact ≤ 60 mm below the margin excluded low-lying GBs entirely; the frozen bounds were checked by dense sampling. |
| axis instability ≤ 10° | Half the alignment tolerance; ρ = λ₁/λ₂ was tried first and rejected (it did not separate unstable axes). |
| f_min | 5th percentile of the target fraction, re-derived after φ was added. |

Not modelled: probe–skin contact quality, refraction and reverberation, partial acoustic transmission.
