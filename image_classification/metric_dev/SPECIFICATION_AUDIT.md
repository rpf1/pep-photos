# Specification coverage — 2026-10-05

Current delivery results: `results/v2`. `results/v1` is retained unchanged. This audit concerns the 2026-10-02 metric-development specification, using already-produced morphology exports. It does not claim access to the live Archie filesystem or raw trajectories.

| Specification area | Evidence | Status / limit |
|---|---|---|
| 0: read-only source audit | AUDIT.md, input_manifest.csv, identity/exclusion/missingness tables | Supplied exports reconciled; live tree unavailable |
| 1: unified physical data | data/unified_metrics.csv, data/DATA_DICTIONARY.md | Raw, signed/absolute deviations and B/M pairs retained; missing 80–100 ns APL explicit |
| 2: three scalings | scaler_parameters.csv, frozen_parameters.json | IQR, null-MAD, q95; zero retained; null failures explicit |
| 3: PCA | pca scores/loadings/parameters/summary/associations | Full/reduced/paired; saved orientation; class FDR added |
| 4: per-membrane effects | effect_scores.csv; supervised_all_comparators.csv | Single metrics, distances, PC1, weighted and supervised controls |
| 5: selectivity mathematics | synthetic grid, metric properties, edge cases, Pareto fronts | Difference, ratios, log ratios, contrast, angle, magnitude, utilities, ideal point |
| Component associations | score_component_correlations.csv; supervised_validation/tables equivalent | Spearman and molecule-bootstrap CIs; no chemical prediction claim |
| Visual nominal classes | class_descriptives/omnibus/pairwise; supervised equivalents | IQR, effect sizes, separately scoped BH-FDR; no ordinal class assumption |
| Classification validation | all_score_grouped_CV.csv; supervised_nested_CV.csv | Train-only transformations and nested supervised regularisation; small null class flagged |
| Ranking uncertainty | score_bootstrap_intervals.csv; supervised_score_bootstrap_intervals.csv | Unsupervised calibration refit; supervised training refit conditional on folds/C |
| Mathematical stress tests | candidate_metric_properties.csv, synthetic_metric_grid.csv, unit tests | Origin, singularities, monotonicity, bounds and weak-effect failure |
| Figure consistency | plot_style.json, plot_manifest.csv, delivery_validation.csv, FIGURE_REVIEW.md | PNG >=300 dpi + SVG, shared axes; reviewed contact sheets |
| Essential figure families A–I | figures/ plus manifest | Reference, preparation QC, temporal/reference, PCA, effect plane, surfaces, selectivity/magnitude, numeric comparisons and all disagreement screens |
| Numeric decision record | tables/decision_table.csv | Effect rows and supervised controls included; missing cells explicitly inapplicable; no opaque winner |
| Score definitions | score_definitions.csv | Frozen formula/scaling/direction/limits; supervised comparator semantics explicit |
| Reproducibility/provenance | analysis_manifest.json, VALIDATION.md, source hashes | v1 computation provenance plus completion provenance; fixed RNG streams |
| Safety and integration | README, CONTINUATION_HANDOVER.md, dry Slurm | No jobs or canonical changes; no SASA or chemical ML training |

## Unavailable rather than unfinished

Raw trajectories/replicate pristine trajectories, locked chemical split/descriptors, independent biological outcomes and live Archie access were not supplied. These prevent temporal-block uncertainty, chemical extrapolation validation, efficacy/toxicity claims and live-environment verification. The legacy Nfe-Nfnap-Nfe identity mismatch stays excluded.

No per-score incremental predictive advantage over ddAPL is claimed from correlation. The demonstrated incremental evidence is the explicitly nested feature-set comparison. Utilities are preferences for later ranking, not validated biological targets.
