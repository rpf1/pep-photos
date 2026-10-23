# PFPML metric development — delivery and continuation

Updated 2026-10-05. The cancelled response did not lose the results. The corrected full run finished successfully on 2026-10-02 (494 seconds). Existing inputs, results and logs are preserved. This is a completed computational analysis with remaining scientific/reporting QA; it is not approval to adopt a canonical ML target.

## Authoritative specification and inputs

Read TASK_SPECIFICATION.md in full before continuing. It is the metric-development specification dated 2026-10-02. The older production-morphology specification is retained in the original uploads; this package develops metrics from already-produced morphology, not trajectory production.

Sources and hashes are recorded in input_manifest.csv and results/v1/analysis_manifest.json. The enriched source master remains outside the package at:
`/workspace/scratch/5750fb2e42a4/presentation_work/inputs/results/production_master_20260930_120836/master_with_morphology.csv`.
Do not substitute an older selected population. Small authoritative selected/pair/reference inputs are included under inputs/.

## Exact analysis population

- Enriched master: 486 rows; 395 production morphology outputs.
- Strict selected dataset: 393 systems = 383 PASS + 10 REVIEW_REQUIRED.
- Paired dataset: 166 molecule/physical-run pairs; 156 pairs with both membranes PASS form primary analysis.
- Two Nfe-Nfnap-Nfe run_0 membrane rows are excluded for unresolved legacy canonical APL run identity. Older presentation's 157 clean pairs are superseded by this strict 156-pair analysis.
- Ten review pairs enter sensitivity analysis only; all ten review systems are bacterial.
- Pristine references: exactly two full-precision rows in baseline_core.csv.
- Only four flat_no_interaction systems per membrane; null scaling is exploratory.
- Late morphology: 80–100 ns medians. No matching 80–100 ns APL; 90–100 ns APL is retained separately, never silently substituted.

## Current stage and results

| Stage | Status |
|---|---|
| Audit and identity reconciliation | Complete for supplied exports; live Archie tree not mounted |
| Unified raw/reference/start/endpoint/late table | Complete: 166 rows, 533 columns; primary inclusion flag |
| IQR/null-MAD/q95 frozen scaling | Complete; unavailable null calibrations explicitly marked |
| Full/reduced/paired PCA | Complete |
| Effect and selectivity candidates / synthetic stress tests | Complete: 1,760 effect/joint score columns |
| Bootstrap / grouped CV | Completed: 300 calibration resamples; 3 CV repeats |
| Figures | 51 figures, PNG and SVG; axis tests pass; visual layout review remains |
| Reporting/integration readiness | Summary, decisions and definitions produced; gaps below remain |

Read results/v1/SUMMARY.md for numerical findings. Morphology improves visual-deformation discrimination over dAPL alone: bacterial OOF AUROC 0.570 to 0.873; mammalian 0.448 to 0.818. Height alone gives 0.900 and 0.830 respectively. These are phenotype comparators, not evidence of antimicrobial efficacy, toxicity or chemical-model predictability.

Endpoint-reference versus endpoint-system-start Spearman ranges 0.9879–1.0000. Late full PCA PC1 explains 88.4% B, 78.7% M and 63.4% paired. Contrast and angle give identical ordering; retain magnitude alongside either. Ratios are diagnostic only. Recommend signed physical responses and separate B/M summaries; no scalar is canonised.

## Validation performed

- All 11 unit tests pass, including generated axis checks.
- Python compilation and Slurm bash syntax pass.
- Full local run completed; manifest records source hashes unchanged.
- Frozen application exercised on all 166 unified rows, reproducing all 1,760 score columns with maximum absolute discrepancy 7.0e-9 (CSV precision) and identical NaN masks. New outputs: results/frozen_reapplication_20261005.csv and provenance JSON.
- No Slurm job submitted, no canonical data/controller/selection altered, no chemical ML training, no SASA.

## Remaining work before claiming every requirement fulfilled

1. Visually inspect figures. Matplotlib emitted tight_layout/colorbar compatibility warnings; programmatic axis tests do not prove unclipped titles/legends.
2. Enrich decision_table.csv with numeric per-score class separation and incremental evidence; some entries currently contain generic interpretation text. Include explicit per-effect recommendations.
3. Supervised logistic comparators have nested grouped OOF validation and conditional bootstrap performance intervals, but do not yet have every unsupervised score's component-correlation/ranking-bootstrap/decision-table coverage. Clearly distinguish these comparator outputs.
4. Audit the full specification requirement-by-requirement, especially figure title/unit semantics and coverage of disagreement cases. Do not equate successful execution with full scientific acceptance.
5. Live Archie source discovery and environment compatibility are untested. Inspect canonical paths read-only before use. Do not auto-repair the historical identity mismatch.
6. No raw trajectories, replicate pristine trajectories, chemical descriptors/locked split, or independent biological outcomes were supplied. Temporal block uncertainty, chemical holdout prediction and biological validation cannot be claimed.

## Resume commands

From the parent of metric_dev, preserving results/v1:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python metric_dev/scripts/workflow.py run --master presentation_work/inputs/results/production_master_20260930_120836/master_with_morphology.csv --selected metric_dev/inputs/current/analysis_dataset_selected.csv --pairs metric_dev/inputs/current/primary_matched_pairs.csv --reference metric_dev/inputs/baseline_core.csv --output metric_dev/results/v2 --bootstrap 300 --cv-repeats 3
python -m unittest discover -s metric_dev/tests -v
```
The archive intentionally omits the large master; use its recorded original source rather than duplicating it. Unified data and frozen calibration are included and sufficient to inspect/reproduce scores.

## Proposed integration / cleanup / Slurm

Install only as a new standalone `analysis/image_classification/metric_dev/` package. Freeze calibration on the permitted training/reference population before later chemical ML; keep B/M targets separate and compute utilities after prediction. Do not modify canonical APL, annotation, active-learning, simulation or morphology-pilot state.

Cleanup approved/executed: none. Proposed destructive cleanup: none. Partial fixture output results/check and Python caches remain locally and are excluded from the delivery archive; no original project file was deleted, moved or renamed.

Exact optional later submission command (NOT EXECUTED; metric-analysis job, not morphology trajectory production):
```bash
cd /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/metric_dev
sbatch slurm/run_metric_analysis.sbatch
```
The script uses my_env_2025, one CPU, 8 GB, 4 hours and project-root read-only discovery. Any production-morphology submission command must come from the separately validated production package; do not infer it from this analysis script.
