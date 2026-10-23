# PFPML metric-development final handover

Use results/v2 for the current delivery. All 13 tests pass; 61 delivery checks pass; 55 PNG/SVG figure pairs were reviewed. Original results/v1 and all source data are preserved. Read TASK_SPECIFICATION.md, SPECIFICATION_AUDIT.md and results/v2/VALIDATION.md before adoption.

## Population and exclusions

156 primary pairs (both membranes PASS), 166 pairs including review for sensitivity; 393 selected membrane systems (383 PASS, 10 REVIEW). Exactly two full-precision pristine references. Exclude both Nfe-Nfnap-Nfe run_0 membrane rows pending canonical APL identity reconciliation. Ten review pairs are sensitivity-only; their review systems are bacterial. No 80–100 ns APL exists, so persistent-response composites contain morphology only.

## Scientific result

Morphology materially improves visual-deformation discrimination over dAPL: bacterial nested OOF AUROC 0.570 to 0.873; mammalian 0.448 to 0.818. Height alone is 0.900/0.830. This validates alignment with structural phenotype, not antimicrobial efficacy or prediction from chemistry.

Recommend signed physical responses and separate B/M effect summaries. Pair contrast (or equivalent angle) with magnitude; use explicit utilities only after prediction. No single biological scalar is approved.

## What was completed after the cancelled output

- Numeric decision evidence added for each applicable candidate and both membranes; effect/control rows included.
- All eight supervised comparators now have component associations, nominal-class tests and 300 training-molecule bootstrap refits, score/rank intervals and top-10 stability. Held-out identities stay excluded; uncertainty is conditional on fixed folds and original nested-selected C.
- PCA exploratory class tests receive BH correction.
- Figures use constrained layout; all disagreement screens and numeric candidate class/rank/Pareto comparisons are plotted.
- Saved-model reproducibility, source hashes, CSV finite-value checks, figure files and axes verified.
- Requirement coverage and unavailable-data boundaries are explicit in SPECIFICATION_AUDIT.md.

## Unified entrypoint

```bash
python scripts/workflow.py complete --source results/v1 --output results/v2 --bootstrap 300
python scripts/workflow.py validate --output results/v2
python -m unittest discover -s tests -v
```
The completion command refuses existing output directories. Use a new version when re-running. The delivery includes v2 and does not need to be rerun merely to inspect results. See README for fresh source-driven runs and frozen application.

## Integration and exact later submission

New standalone destination:
`/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/metric_dev/`

Before chemical ML, fit/freeze calibration on the permitted training/reference population and validate using the locked split. Live Archie source discovery/environment requires read-only verification. No chemical model training or active-learning selection was performed.

Optional later metric-analysis submission command, NOT EXECUTED:
```bash
cd /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/metric_dev
sbatch slurm/run_metric_analysis.sbatch
```
This runs metric analysis, not morphology trajectory production. Do not use it as a production morphology submission command.

## Cleanup

No destructive cleanup proposed or executed. No original project files were deleted, moved or renamed. Delivery excludes local fixture outputs/caches and duplicate downloaded archives without deleting them. v1 and v2 are retained for provenance. Large source master remains at its manifest path and is not unnecessarily duplicated.

## Remaining external validation

Trajectory-block uncertainty, replicated pristine null, chemical predictability, biological efficacy/toxicity and Archie compatibility require unavailable data/access. No package result establishes these. Original interrupted-state CONTINUATION_HANDOVER.md is retained as historical context; this document supersedes its unfinished computational QA list.

The new archive contains the current v2 analysis. Original v1 remains in the workspace and earlier delivery archive, avoiding redundant bulk copies. To run completion again from v1, retain/extract that earlier archive.
