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

## 2026-10-05 figure-layout migration

Phase 0 discovered 55 logical figures, 110 format copies and nine families. Legacy consumers were Plotter.save/finish, completion validation, and v2-only tests. The existing annotation GUI provides atomic fsynced state writing, aspect-preserving Pillow display, viewport rendering and typing-safe navigation patterns.

Phase 1 uses format-first figures/png and figures/svg paths and one logical manifest row, preserving every previous metadata field except the replaced per-format filename. Explicit workflow migrate-figures preflights all pairs, refuses conflicts, preserves hashes, and rolls back on failure. figure_layout.py is a shared durable contract used by plotting validation, migration and the future reviewer. Current migration passed file, identity, hash and axis checks before guide or GUI implementation. Scientific data and tables were unchanged; exact evidence is in figure_layout_validation.json. Historical v1/v2 handovers are earlier stages, not current Archie status.

## 2026-10-05 guided review architecture

- The Phase-1 gate passed before constructing the guide or GUI: 55 logical figures, 110 unchanged image hashes, nine families, unchanged protected scientific files, and 13 scientific tests.
- One human-readable `figure_guide.json` avoids per-image prose files. It separates reusable metric/technique definitions from figure-specific observations and precise table filters. Coverage validation makes missing explanatory fields an error.
- Original hypotheses have independent design provenance. Most comparisons are explicitly exploratory. Present-day reading criteria are labelled separately; observed extrema are descriptive table summaries. Fresh Archie outcomes supersede older v1/v2 numerical summaries without rewriting historical handovers.
- tkinter/ttk + Pillow follows the existing validated annotation-GUI approach. Atomic fsynced replacement, typing-safe navigation, aspect-preserving fit and viewport-only rendering are reused conceptually. No code or annotation state in pfp34_v2 was modified.
- `review_metric_figures.py` is the durable review entrypoint. It exposes detailed separate Design and Result tabs, axes, metric formulas, limitations, related figures and read-only source previews.
- Personal status/notes belong only in `figure_review_state.json`, never in authoritative scientific metadata. Result identity is tied to the analysis-manifest hash, independent of installation path. Linux directory locking and optimistic content checks guard concurrent edits; notes are saved immediately.
- Existing report-generation statements asserting an untested Archie environment were replaced with execution-context-neutral wording. Current SUMMARY/VALIDATION identify the supplied successful job; the original analysis manifest and historical documents remain unchanged. The completion validator no longer fabricates source_result=results/v1 during validation.
- No results were recomputed, no Slurm work was submitted, and no canonical analysis/annotation/active-learning state was accessed. Existing scientific tables, including delivery_validation.csv and score_definitions.csv, remain byte-identical.
- Current validation: 25 tests and headless review checks passed. The environment lacks DISPLAY/Xvfb; rendered GUI and ThinLinc SVG-handler behaviour remain unverified. This limitation is explicit in the delivery rather than inferred from headless success.
