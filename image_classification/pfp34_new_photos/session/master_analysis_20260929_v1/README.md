# PFPeptoids canonical manual + simulation master analysis

## Main result

This directory was generated from the canonical annotation export and
`analysis/masterdataanalysis.csv` without altering either source.

The main requested three-row representation is:

1. `mammalian` = `POPC60CHOL40_W_WF`
2. `bacterial` = `POPE75POPG25_W_WF`
3. `bacterial_minus_mammalian`

`ddAPL = ΔAPL_bacterial - ΔAPL_mammalian`.

The third row does **not** subtract categorical labels or species-specific
local-lipid measurements. It contains the two source labels in
`manual_label_mammalian` and `manual_label_bacterial`; numeric subtraction is
restricted to metrics listed as allowed in `08_metric_dictionary.csv`.

## Primary dataset

`05_tripeptoid_three_row_primary.csv` is the first analysis dataset.

A primary pair requires:
- peptoid length = 3;
- both mammalian and bacterial physical simulations;
- canonical 100 ns simulation analysis complete;
- both manual annotations `labelled`;
- both supported by `STACKED` evidence.

`POPC80POPS20_W_WF` is retained separately in `07_popc80pops20_systems.csv` and is never
silently pooled with either member of the main pair.

## Morphology

The main morphology columns use grid 32 / 4 nm smoothing from the validated
pilot. PO4 is a geometry marker only. Canonical APL remains BOX XY projected
area per membrane molecule.

Redundant morphology measures are not promoted into the core three-row table.
The retained core is height RMS, RMS mean curvature, midplane excess area and
median local-normal thickness, with baseline/endpoint/delta, temporal summaries
and QC where available.

## Local lipid state

Local-lipid outputs are retained in `01_system_master_extended.csv` and support
tables, but are explicitly marked as pilot/non-production. Leaflets are paired
parts of the same simulation, not independent observations.

CHOL near-midplane and assignment-change QC is preserved. Dense CHOL sampling
overlaps the smoke pilot and is not counted as an additional replicate.

## Annotation concentration ambiguity

Any annotation record containing evidence from more than one `nmol` is never
silently copied onto individual simulation concentrations. Such rows are
isolated in `10_annotation_concentration_ambiguities.csv`.

## Files to use first

- `02_system_master_core.csv`
- `05_tripeptoid_three_row_primary.csv`
- `06_tripeptoid_pairs_wide_primary.csv`
- `14_label_counts.csv`
- `15_pair_label_counts.csv`
- `16_ddapl_by_label_pair.csv`
- `17_metric_summary_by_label.csv`
- `18_numeric_correlations.csv`
- `19_target_candidate_summary.csv`
- `VALIDATION.json`

No composite score or final ML target is chosen by this script.
