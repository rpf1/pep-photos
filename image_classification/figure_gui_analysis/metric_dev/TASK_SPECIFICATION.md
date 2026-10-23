# PFPML Work Task — Develop a reference-normalised membrane-effect/selectivity metric

## 1. Objective

Develop and validate a **continuous, interpretable membrane-effect framework** that improves on using `dAPL(B)`, `dAPL(M)` and `ddAPL = dAPL(B) - dAPL(M)` alone.

The end goal is to determine whether we can define:

- `Effect_B`: continuous bacterial-membrane effect
- `Effect_M`: continuous mammalian-membrane effect
- `Selectivity`: which membrane is preferentially affected
- optionally one scalar `Desirability`: poor candidate -> desirable bacteria-selective candidate

The priority is **not** to force one score. The priority is to find the simplest physically defensible representation that avoids the information-loss problem in ddAPL, then determine whether a single scalar can be justified.

A key failure mode to solve is:

- B ~ 0 and M ~ 0 -> difference ~ 0
- B large and M large -> difference can also ~ 0

Those are physically different states but are collapsed by a simple difference.

Raw ratios also have a failure mode:

- `B/M` explodes as `M -> 0`, potentially making a weak bacterial effect appear excellent simply because mammalian effect is extremely small.

Investigate these mathematically and empirically rather than assuming a preferred formulation.

## 2. Scientific definitions that are fixed

### Membranes

- bacterial-like: `POPE75POPG25_W_WF`
- mammalian-like: `POPC60CHOL40_W_WF`

### Canonical projected-area response

- APL = projected box XY area / membrane molecules per leaflet
- dAPL = APL at Relax 100 ns - APL at beginning/reference Pull frame
- ddAPL = dAPL_bacterial - dAPL_mammalian

### State terminology

Use these names consistently:

- **pristine reference** = the two peptoid-free `prod.gro` bilayers
- **system start** = first Pull-frame state of an individual peptoid-containing simulation
- **endpoint** = production endpoint
- **late window** = late-time morphology summary, preferably 80–100 ns if available

Do **not** call system start “baseline” in new outputs because “baseline” became ambiguous.

### Pristine-reference source structures

Exactly two pristine reference boxes exist:

`/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Reference/2.1/bilayers/POPE75POPG25_W_WF/prod.gro`

`/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Reference/2.1/bilayers/POPC60CHOL40_W_WF/prod.gro`

The completed reference morphology analysis reported approximately:

- POPE75POPG25: APL 59.079688 A^2; height RMS 0.098982 nm; excess-area fraction 0.003082; RMS curvature 0.040531 nm^-1; thickness 4.067303 nm
- POPC60CHOL40: APL 44.465312 A^2; height RMS 0.210445 nm; excess-area fraction 0.010899; RMS curvature 0.072522 nm^-1; thickness 4.132899 nm

Do not hard-code these rounded numbers if the canonical `baseline_core.csv` is available. Read the full-precision values from the file.

## 3. Important SASA decision

SASA is deliberately **not part of the primary metric-development workflow**.

Reason:
- the morphology question is directly addressed by geometric surface area/excess area, height RMS, curvature, thickness and leaflet asymmetry;
- Martini coarse-grained SASA additionally requires a clearly frozen bead-radius/probe convention;
- adding SASA before that convention is agreed would introduce a method-dependent quantity without clear evidence that it adds independent information.

Preserve this decision in `DECISIONS.md`.

Do not calculate SASA in this task.

## 4. Existing evidence to preserve

The validated morphology core is:

1. height RMS
2. excess surface area
3. RMS mean curvature
4. median local-normal thickness
5. leaflet asymmetry measures

Previous analyses showed strong redundancy between some morphology metrics, especially height RMS / excess area / curvature. Do not silently drop them: quantify redundancy again in the unified dataset and compare a full feature set against a reduced feature set.

The pristine-reference comparison also indicated that `endpoint - system_start` and `endpoint - pristine_reference` are extremely strongly correlated for the existing global morphology measures. Treat this as something to verify from source data, not as a hard-coded fact.

Manual visual classes currently include:

- `flat_no_interaction`
- `flat_interaction`
- `bulge_flat`
- `curved`
- `bulge_curved`
- `pore_formed` where present

Important:
- do not silently pretend these are all a single perfectly ordered biological-response scale;
- `flat_interaction` vs `flat_no_interaction` explicitly encodes interaction;
- curvature/bulging classes primarily encode morphology;
- use the classes as validation/phenotype evidence, not unquestioned ground truth for a single ordinal score.

## 5. Safety / provenance

Start with a read-only audit.

Do not:
- modify canonical APL outputs;
- modify `analysis/masterdataanalysis.csv`;
- modify active-learning databases/controllers;
- rewrite existing morphology production results;
- submit Slurm jobs;
- delete/move/rename existing files.

You may create a new standalone analysis package and dry-run Slurm script.

Proposed short package root:

`/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/metric_dev/`

Keep paths short.

Do not duplicate large source datasets unnecessarily. Record source paths + SHA256 hashes in a manifest.

## 6. Checkpoint 0 — Audit and identify authoritative inputs

Before writing analysis logic, inspect the current live tree and identify the authoritative input files.

Likely locations include:

`analysis/image_classification/flatness_physics_analysis/`

Look for the newest/current equivalents of:
- `primary_matched_pairs.csv`
- `analysis_master.csv`
- `analysis_bundle.xlsx`
- `analysis_dataset_selected.csv`
- any late-window morphology tables

Pristine reference result should preferably be:

`analysis/image_classification/morph_prod/results/baselines/prod_20261001/baseline_core.csv`

If housekeeping has not yet moved it, legacy fallback may be:

`analysis/image_classification/morph_prod/results/reference_baselines/prod_gro_final_20261001/baseline_core.csv`

Use only one authoritative pristine-reference file and record why.

Audit:
- row counts
- unique molecule / physical-run identities
- B/M pair completeness
- label counts by membrane
- QC status
- units
- available endpoint, system-start, delta, late-window and asymmetry columns
- duplicate identities
- missing values

Create:

`metric_dev/AUDIT.md`

and

`metric_dev/input_manifest.csv`

with absolute source path, role, SHA256, rows, columns, selection reason.

Do not proceed silently if identities cannot be reconciled.

## 7. Checkpoint 1 — Build one unified metric-development table

Create one canonical analysis table at the paired molecule/run level.

For every physical metric X and each membrane, retain the raw values and derive:

### A. Pristine-reference deviation

`endpoint_vs_ref = endpoint - pristine_reference`

This answers: “How different is the final membrane from the untouched membrane?”

### B. Late-window pristine-reference deviation

`late_vs_ref = late_window - pristine_reference`

Prefer the 80–100 ns median/mean that is already produced by the morphology workflow, if available.

This should be treated as the main persistent-response candidate because it is less sensitive to a single endpoint frame.

### C. Simulation change

`endpoint_vs_start = endpoint - system_start`

Retain the existing production delta. This answers: “What changed during the peptoid-containing simulation?”

### D. Start offset / preparation QC

`start_vs_ref = system_start - pristine_reference`

This is primarily QC/mechanistic context: “Was the membrane already globally perturbed at the system-start state?”

### E. Cross-membrane paired quantities

For corresponding B and M measures create:
- `B_minus_M`
- `abs_B_minus_M`
- B and M retained separately

Do not use raw ratios at this stage.

For sign-sensitive features, preserve signed and absolute deviations separately rather than silently converting everything to magnitude.

Save:

`results/v1/data/unified_metrics.csv`

Include a data dictionary:

`results/v1/data/DATA_DICTIONARY.md`

## 8. Checkpoint 2 — Define candidate per-membrane effect features

The goal is to derive **non-negative membrane-effect magnitude** for each membrane before building selectivity.

Test multiple ways of scaling the heterogeneous physical features.

At minimum compare:

### Scaling A — Standard robust scaling

Reference-normalised feature:
`x = endpoint_or_late - pristine_reference`

Scale each feature with a robust scale such as median/IQR or MAD derived from the analysis population.

Keep the pristine point explicitly at zero before scaling.

### Scaling B — Null/no-interaction scaling

Where statistically defensible, compare against the `flat_no_interaction` distribution.

However the no-interaction group is small. Do not make hard thresholds from it without uncertainty/bootstrap analysis.

### Scaling C — Robust percentile / q95 scale

Compare scaling by a robust high quantile of absolute deviations.

Freeze every scaler parameter and save it so a future molecule can be transformed identically.

Create:

`results/v1/data/scaler_parameters.csv`

Do not choose one scaling method only because it produces prettier separation.

## 9. Checkpoint 3 — PCA / dimensional structure

Run PCA as an **exploratory physics analysis**, not automatically as the final target.

Run at least:

### PCA-A: full reference-normalised morphology + APL set
For each membrane separately.

### PCA-B: reduced/core feature set
Use redundancy analysis to remove near-duplicate features, but document exactly what was removed and why.

### PCA-C: paired B+M representation
One row per molecule/run with both bacterial and mammalian features.

All PCA inputs must be scaled appropriately because units differ.

Report:
- explained variance
- cumulative explained variance
- loadings
- scores
- contribution of each physical metric
- sensitivity to full vs reduced feature set
- association with visual classes
- association with dAPL/ddAPL

Do not silently orient PC signs. If an orientation is flipped for interpretation, record it.

Outputs:

`results/v1/data/pca_scores.csv`
`results/v1/data/pca_loadings.csv`
`results/v1/tables/pca_summary.csv`

## 10. Checkpoint 4 — Candidate membrane-effect scores

For each membrane independently, compare simple and interpretable candidates.

At minimum:

1. single-metric controls: dAPL, height RMS vs pristine, excess area vs pristine, curvature vs pristine, thickness vs pristine, leaflet asymmetry
2. Euclidean distance from pristine in a scaled core-feature space
3. robust distance from pristine such as covariance-aware / shrinkage Mahalanobis if numerically stable
4. PCA-derived effect, e.g. oriented PC1 only if loadings and class behavior are physically coherent
5. simple linear combination with weights derived transparently from the data
6. supervised linear candidate only as a comparator, using leakage-safe cross-validation and explicit regularisation

Do not use black-box models as the primary score-building method. A score intended to become an ML target needs to remain interpretable.

Create `Effect_B_*` and `Effect_M_*` for every candidate family.

## 11. Checkpoint 5 — Selectivity mathematics

Once a non-negative `Effect_B >= 0` and `Effect_M >= 0` exist, investigate methods for combining them.

Evaluate these candidate selectivity formulations:

### Difference
`S_diff = B - M`
Known issue: both-low and both-high can collapse to similar values.

### Raw ratio
`S_ratio_BM = B / M`
`S_ratio_MB = M / B`
These are diagnostic comparators only. Quantify denominator instability and singular behavior.

### Stabilised/log ratio
`log((B + c) / (M + c))`
Do not select an arbitrary pseudocount silently. Test sensitivity over defensible values of `c`, including a noise-floor-based value if justified.

### Normalised contrast
`S_norm = (B - M) / (B + M)`
Define `S_norm = 0` when `B = M = 0`.

Properties:
- bounded [-1, +1]
- +1 = bacterial-only effect
- 0 = equal effect
- -1 = mammalian-only effect

### Angular selectivity
For B,M >= 0:

`theta = atan2(B, M)`

Rescale to:

`S_angle = (4/pi) * atan2(B, M) - 1`

Properties:
- bounded [-1, +1]
- stable when one effect is zero
- equivalent to representing the B/M relationship as direction rather than an exploding raw ratio

### Effect magnitude
`Magnitude = sqrt(B^2 + M^2)`
This is not selectivity. It answers how much total membrane response occurred.

### Desirability / utility
After putting B and M on a fixed interpretable scale:
- bacterial desirability increases with B
- mammalian safety increases as M decreases

Test weighted desirability formulations, including geometric-mean style desirability. Do not hide the weights. Compare several reasonable weight choices.

### Ideal-point distance
Define ideal bacteria-selective behavior as high B / low M in a normalised space and measure distance to that point.

### Pareto analysis
Identify Pareto-dominated and non-dominated molecules using:
- maximize B
- minimize M

Pareto analysis is the no-information-loss benchmark against which scalar rankings should be checked.

## 12. Metric stress tests

This is mandatory.

Before choosing any scalar score, evaluate it mathematically over a synthetic B/M grid spanning the valid normalised effect range.

The purpose is to show:
- behavior at B=M=0
- behavior when both are high
- behavior as M -> 0
- behavior as B -> 0
- monotonicity
- boundedness
- singularities
- whether a weak B effect can score artificially well solely because M is tiny

Create:

`results/v1/tables/candidate_metric_properties.csv`

Columns should include at least:
- metric name
- formula
- range
- bounded?
- singularity?
- zero/zero behavior
- monotonic in B?
- monotonic penalty in M?
- Pareto-consistent?
- arbitrary constant required?
- interpretation
- principal failure mode

This is a major decision table.

## 13. Statistical evaluation

For every candidate effect/selectivity/desirability score:

### Continuous comparisons
- Spearman correlations with component metrics
- redundancy/correlation matrix
- bootstrap confidence intervals

### Visual phenotype comparisons
Compare score distributions across:
- flat_no_interaction
- flat_interaction
- bulge_flat
- curved
- bulge_curved

Use:
- medians and IQR
- Mann–Whitney for planned pairwise comparisons
- BH-FDR correction
- rank-biserial effect size
- Kruskal–Wallis overall where appropriate

Do not rely on p-values alone.

### Classification-style validation
Where appropriate:
- one-vs-rest AUROC
- balanced accuracy or equivalent for simple threshold-based discrimination
- cross-validation grouped by molecule when repeated identities/runs could leak

The small `flat_no_interaction` class must be highlighted as a limitation.

### Ranking stability
Bootstrap molecules and report:
- score confidence intervals
- top-k rank stability
- Spearman rank agreement across bootstrap samples

### Pareto consistency
If molecule A has B_A >= B_B and M_A <= M_B with at least one strict inequality, then a “better is larger” bacteria-selective scalar should not rank A below B unless there is a documented reason.

Count and report violations.

## 14. Figure system — strict consistency

Figures must be publication/supervisor ready and highly systematic.

Create one plotting configuration/module. Do not scatter style constants across scripts.

Use consistent:
- font family
- title size
- subtitle style if used
- axis-label size
- tick-label size
- legend size
- marker sizes
- line widths
- alpha
- figure width/height
- DPI
- padding
- naming convention

Prefer vector output plus PNG:
- `.png` at >=300 dpi
- `.svg` or `.pdf`

### Shared axis rules

For B/M counterpart plots of the same metric:
- use the **same x/y limits**
- use the same tick spacing where practical
- use the same visual encoding
- state units identically

For B-vs-M scatter plots:
- same numeric range on x and y
- equal aspect ratio where meaningful
- include x=y reference line

For selectivity:
- always use the same `[-1, +1]` axis when the score is bounded to that range

For 0–1 effect/desirability scores:
- always use `[0, 1]`

For PCA counterpart panels:
- use common PC1/PC2 limits when comparing the same PCA representation/classes

For class-distribution plots of the same metric:
- use one shared y-range across membrane panels

Do **not** force unrelated metrics with different physical units onto one identical numeric axis.

### Clear figure naming

Use numbered, descriptive filenames.

Suggested structure:

`results/v1/figures/01_reference_deviation/`
`results/v1/figures/02_class_comparisons/`
`results/v1/figures/03_redundancy/`
`results/v1/figures/04_pca/`
`results/v1/figures/05_metric_surfaces/`
`results/v1/figures/06_effect_scores/`
`results/v1/figures/07_selectivity/`
`results/v1/figures/08_disagreement_cases/`
`results/v1/figures/09_bootstrap/`

Examples:

`01_height_rms_endpoint_vs_ref_by_class.png`
`02_excess_area_endpoint_vs_ref_by_class.png`
`03_B_vs_M_height_rms_ref_deviation.png`
`04_PCA_full_PC1_PC2_by_visual_class.png`
`05_metric_surface_raw_ratio_B_over_M.png`
`06_metric_surface_normalised_contrast.png`
`07_metric_surface_angular_selectivity.png`
`08_effectB_vs_effectM_by_class.png`
`09_selectivity_vs_magnitude.png`
`10_desirability_vs_ddAPL.png`

Every figure title should say exactly:
- what quantity is shown
- which membrane(s)
- whether it is endpoint-vs-reference, late-vs-reference, or endpoint-vs-start
- units

Avoid vague titles such as “comparison” or “results”.

## 15. Essential figure families

At minimum produce:

### A. Pristine reference vs system classes
For each core metric:
- pristine reference horizontal line
- distributions of endpoint/reference deviation by visual class
- B and M shown with consistent scales

### B. System-start QC
For each core metric:
- `system_start - pristine_reference`
- compact plot showing the distribution is near/not near zero
- this is QC, not the headline result

### C. Endpoint-vs-reference versus endpoint-vs-start
Scatter for each metric:
- x = endpoint - system start
- y = endpoint - pristine reference
- x=y line
- report Spearman
- shared scale

### D. PCA
- scree plot
- PC1/PC2 by class
- loadings
- B vs M counterpart views
- paired PCA if useful

### E. B/M effect plane
Core plot:
- x = Effect_M
- y = Effect_B
- equal scales
- x=y line
- visual class encoded
- annotate only a small set of scientifically interesting outliers

### F. Metric surfaces
For difference, raw ratio, log-ratio, normalised contrast, angular selectivity, desirability:
- same B and M grid
- clear colorbar
- show singular/capped regions explicitly
- make the ratio pathology visually obvious

### G. Magnitude vs selectivity
- x = selectivity
- y = magnitude
- this plot must demonstrate why equal selectivity can correspond to weak or strong total response

### H. Candidate-score comparison
- class separation
- ddAPL comparison
- rank agreement
- Pareto violations
- bootstrap stability

### I. Disagreement cases
Identify examples where:
- ddAPL ~ 0 but morphology effect is high
- ddAPL large but morphology effect is low
- high selectivity but low magnitude
- high magnitude but low selectivity
- score candidates disagree strongly

Create a table and labelled plots for these cases.

## 16. Agent-readable outputs

The package must be understandable without opening figures manually.

Create:

`metric_dev/README.md`
- what the package does
- exact run command
- input expectations
- output map

`metric_dev/DECISIONS.md`
- pristine-reference terminology
- system-start terminology
- SASA exclusion and reason
- methods tested
- methods rejected
- no final metric canonised unless evidence supports it

`results/v1/SUMMARY.md`
- concise scientific results
- strongest trends
- whether pristine-reference changes conclusions
- PCA interpretation
- score comparison
- recommended shortlist
- limitations
- exact next decision

`results/v1/analysis_manifest.json`
- timestamp
- source hashes
- script hashes
- parameters
- versions
- row counts
- exclusions

`results/v1/plot_manifest.csv`
Columns:
- figure_id
- filename
- title
- question
- x_variable
- y_variable
- grouping/color variable
- units
- axis limits
- source table
- one-sentence interpretation

`results/v1/score_definitions.csv`
Columns:
- score_name
- formula
- required inputs
- scaling
- direction
- range
- interpretation
- limitations
- status: exploratory / shortlisted / rejected

`results/v1/tables/decision_table.csv`
One row per candidate final metric with:
- simplicity
- interpretability
- class separation
- bootstrap stability
- Pareto consistency
- denominator instability
- dependence on arbitrary constants
- redundancy with ddAPL
- added information beyond ddAPL
- recommendation status

Do **not** reduce this table to one opaque “overall score”.

## 17. Recommended package implementation

Prefer a small standalone package:

`metric_dev/scripts/build_metric_analysis.py`
`metric_dev/scripts/plot_metric_analysis.py` if separation is useful
`metric_dev/slurm/run_metric_analysis.sbatch`
`metric_dev/tests/test_metric_analysis.py`

The Slurm job should be CPU-only unless evidence shows otherwise.

Do not submit it.

Use the existing Archie Python environment if compatible:
`my_env_2025`

For numerical libraries on one Slurm task:
- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

Parallelise only if justified by actual workload.

## 18. Validation requirements

Before declaring success:

1. Input identity reconciliation passes.
2. Pristine-reference rows = exactly 2.
3. Every B/M pair maps to the correct pristine membrane.
4. Units are explicit.
5. No divide-by-zero silently produces inf.
6. Raw-ratio singularities are deliberately captured and reported.
7. Score formulas reproduce from saved scaler/weight files.
8. Figure axis rules are programmatically tested where practical.
9. All generated CSVs have deterministic column order.
10. Same random seed used for bootstrap/CV.
11. No source/canonical files modified.
12. Python compilation/tests pass.
13. Slurm script passes `bash -n`.
14. Dry-run or small local fixture test succeeds.
15. No Slurm jobs submitted.

## 19. What NOT to do

- Do not decide in advance that raw ratio, PCA, desirability or any other method is “the winner”.
- Do not train a black-box model and use its prediction as the new physical target.
- Do not merge bacterial and mammalian responses before preserving them individually.
- Do not use a ratio with an arbitrary epsilon and hide the epsilon.
- Do not use only p-values.
- Do not define hard thresholds from the tiny `flat_no_interaction` set without uncertainty.
- Do not call the first Pull frame “baseline” in new outputs.
- Do not calculate SASA in this task.
- Do not overwrite previous analysis.
- Do not submit Slurm work.

## 20. Final scientific decision we want to reach

Determine whether the data support a framework approximately like:

`Effect_B` = continuous bacterial membrane-effect magnitude

`Effect_M` = continuous mammalian membrane-effect magnitude

`Magnitude = sqrt(Effect_B^2 + Effect_M^2)`

`Selectivity` = a stable bounded directional measure such as normalised contrast or angular selectivity

`Desirability` = optional one-dimensional bacteria-selective ranking utility

The strongest candidate should:
- retain the distinction between “nothing happens” and “both membranes strongly affected”;
- reward strong bacterial effect;
- penalise mammalian effect;
- not explode when one effect approaches zero;
- be monotonic and Pareto-consistent;
- be stable under bootstrap;
- remain physically interpretable;
- add information beyond ddAPL;
- be simple enough to freeze as a future ML target.

If no single scalar satisfies those requirements, explicitly recommend keeping `Effect_B` and `Effect_M` as the primary ML targets and using selectivity/desirability only for ranking after prediction.

## 21. Final deliverables

Return:

1. read-only audit
2. complete new package under `metric_dev/`
3. reproducible analysis scripts
4. tests
5. dry-run Slurm script
6. unified reference-normalised dataset
7. PCA outputs
8. candidate effect/selectivity/desirability outputs
9. mathematical stress-test outputs
10. publication/supervisor-ready figures with consistent styling/scales
11. plot manifest
12. score-definition table
13. decision table
14. concise `SUMMARY.md`
15. `DECISIONS.md`
16. exact command the user should run on Archie
17. explicit statement that no Slurm job was submitted
