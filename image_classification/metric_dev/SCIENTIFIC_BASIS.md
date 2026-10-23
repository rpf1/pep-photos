# Scientific basis and ML-readiness boundaries

Reviewed 2 October 2026. These sources guide validation design; they do not establish that the present scores predict antibacterial activity.

| Source | Relevant principle | Application here |
|---|---|---|
| [DataSAIL, Nature Communications (2025)](https://www.nature.com/articles/s41467-025-58606-8) | Identity-disjoint splits can still leak chemical similarity; similarity-aware splits test harder generalisation. | Molecule groups prevent identity leakage here. A chemical-similarity holdout remains a required later gate because molecular structures/descriptors and the locked ML split were not supplied. |
| [Molecular deep learning at the edge of chemical space, Nature Machine Intelligence (2026)](https://www.nature.com/articles/s42256-026-01216-w) | Reliability outside training chemical space needs explicit assessment. | Do not infer novel-peptoid predictability from visual-phenotype discrimination. Retain applicability-domain evaluation for the future chemistry model; no black-box target is introduced. |
| [Guiding questions to avoid data leakage in biological ML, Nature Methods (2024)](https://www.nature.com/articles/s41592-024-02362-y) | Data preparation, split choice and target construction can create leakage. | Refit scales, feature selection and score parameters within folds; nested regularisation selection; preserve B/M pairs as one molecule group. |
| [Scikit-learn: common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Fit preprocessing only on training observations. | The descriptive full-cohort calibration is not used as a prefit transform inside OOF validation. |
| [Scikit-learn: nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) | Model selection and performance estimation need separation. | Inner folds select C; outer molecule groups estimate phenotype discrimination. |
| [Scikit-learn: Ledoit–Wolf](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html) | Covariance shrinkage improves conditioning in correlated feature spaces. | Estimate covariance from winsorised scaled training data; evaluate quadratic distance from pristine zero, not from the fitted cohort mean. This is a regularised distance, not proof of a null distribution. |
| [Best Practices for Quantification of Uncertainty and Sampling Quality in Molecular Simulations (2018)](https://pubmed.ncbi.nlm.nih.gov/30533602/) | Simulation uncertainty depends on sampling quality and correlated observations. | Molecule bootstrap measures cohort/calibration uncertainty. Raw trajectories and replicated reference simulations are needed for correlation-aware block uncertainty and effective sample size. |

## What the available data can establish

1. Mathematical behaviour, singularities, monotonicity and Pareto consistency.
2. Reference normalisation, morphology redundancy and calibration dependence.
3. Alignment with nominal visual phenotypes, with molecule-grouped validation.
4. Ranking sensitivity to cohort resampling, scalers, feature choice, time summary and QC inclusion.

## What remains unestablished

- **Physical repeatability:** independent seeds, longer trajectories, time correlations and pristine-reference variability.
- **Method robustness:** full-population refits at different Fourier cutoffs/grids and marker/leaflet choices.
- **Mechanistic coverage:** permeability, water defects, persistent pores, leaflet mixing, lipid loss and local perturbations can be missed by global geometry. No raw trajectories were supplied to compute them. Such readouts require separate conventions and validation.
- **Biological validity:** membrane structural deviation is not itself antibacterial efficacy or mammalian toxicity. External assays are needed before utility weights acquire biological meaning.
- **Chemical predictability:** these are candidate outputs of future models. Predicting a simulation-derived score from its own component measurements would be tautological. Evaluate predictions from permitted chemical inputs only, using a frozen target recipe and an untouched/similarity-aware test set.
- **Kinetic collective variables:** time-lagged methods such as tICA/VAMP or modern learned collective variables require frame-resolved trajectories and address dynamics. PCA of summaries cannot substitute for that evidence. No kinetic claim is made.

## Freezing a target for ML

Keep the signed physical vector and membrane identities. Fit target calibration on the allowed training/reference population, freeze reference values, feature order, units, scales, weights, transforms and version, then transform held-out/new systems without refitting. Quantify both target measurement noise and chemical-model prediction error. Compare single- and multi-output models under the same locked split. Apply explicit selectivity/desirability after B/M prediction until a scalar has independent validation.
