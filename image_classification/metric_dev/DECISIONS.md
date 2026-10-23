# Decisions

- The two peptoid-free prod.gro bilayers are **pristine references**. First Pull frames are **system start**. Existing source field names are mapped explicitly, never relabelled as pristine measurements.
- SASA is excluded. Geometry directly addresses bending, whereas Martini SASA additionally needs a frozen bead-radius/probe convention and evidence of incremental value.
- The strict 156-pair source is authoritative for this snapshot. The Nfe-Nfnap-Nfe legacy-run mismatch is not automatically repaired. REVIEW_REQUIRED systems enter sensitivity only.
- Keep every source metric and signed/absolute deviation. Compare full/reduced spaces using |Spearman rho| ≥ 0.90 and the documented physical retention order; freeze the selected indices and report removals.
- Main late effects use 80–100 ns morphology. Endpoint candidates may include APL. Missing 80–100 ns APL is not filled with its 90–100 ns mean or endpoint.
- Compare shared B/M IQR, no-interaction MAD and q95 scales. Zero stays at the pristine point. Reject degenerate scales explicitly rather than hiding epsilon terms. Null scaling has only eight pooled no-interaction observations and is not a validated noise floor.
- Compare Euclidean, winsorised Ledoit–Wolf covariance distance, inverse-redundancy weighted absolute sums, absolute PC1, single metrics and directional comparators. Shrinkage regularises covariance; it does not establish a biological null. Mahalanobis may downweight common coherent response and is not automatically preferable.
- Bound distance summaries with d/(1+d), a disclosed monotonic display/utility scale. Report all calibration parameters. Do not interpret these as probabilities.
- PCA is exploratory. Largest absolute loading is oriented positive and each multiplier is saved. Absolute PC1 is a comparator, not an endorsed response axis.
- Supervised L2 logistic candidates are phenotype comparators only. They do not have a pristine-zero interpretation and are not combined into the primary selectivity families. Their validation is nested and molecule-grouped.
- Contrast and angle are equivalent directional rankings. Neither identifies response strength. Keep magnitude and B/M effects. Raw ratios are diagnostic-only and rejected as standalone targets.
- Log ratios use c=0.01, 0.1 and 0.5 on the disclosed bounded scale. These are sensitivity values, not noise estimates. No physical noise-floor pseudocount is justified by one pristine frame.
- Geometric, linear and ideal-point utilities use B weights 0.25, 0.5 and 0.75. No preference is hidden. Pareto analysis is the benchmark; magnitude and inverse ratio are not better-is-larger bacterial utilities.
- Classes are nominal validation evidence. Deformed versus plain-flat and bulge-related contrasts are explicitly defined. No single ordinal label mapping is assumed.
- Bootstrap molecules, not membrane rows. Calibration confidence is separate from trajectory uncertainty. Learn every scaler/threshold/feature selection within validation folds.
- No metric is canonised. Until independent physical/biological and predictive validation exists, retain separate per-membrane physical targets and apply utilities after prediction.
