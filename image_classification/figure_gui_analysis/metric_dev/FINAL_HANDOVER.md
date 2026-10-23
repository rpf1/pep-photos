# Current handover — figure review, 2026-10-05

Current result: `results/archie_8193723_complete`. The historical handover below is retained verbatim and describes the earlier v2 stage, not the current result or current Archie status.

Completed in required order: read-only audit → validated migration → validated guide → tkinter/Pillow reviewer → tests/invariance checks. All 55 logical figures now have PNG and SVG copies under `figures/png/<family>/` and `figures/svg/<family>/`. One logical manifest row retains all former metadata with explicit format paths and hashes. No legacy image tree remains. All 110 image hashes and 59 protected scientific-file hashes are unchanged.

Created: `scripts/figure_layout.py`, `scripts/review_metric_figures.py`, `tests/test_figure_review.py`, and current-result `figure_guide.json`, `figure_layout_validation.json`, `figure_review_validation.json`. Personal `figure_review_state.json` is created on first GUI use only.

Modified: `scripts/plot_metric_analysis.py`, `scripts/complete_metric_analysis.py`, `scripts/report_metric_analysis.py`, `scripts/workflow.py`, `tests/test_metric_analysis.py`, `README.md`, `DECISIONS.md`, this handover, and current-result `plot_manifest.csv`, `SUMMARY.md`, `VALIDATION.md`. Scientific CSV/JSON tables, original analysis provenance, historical design documents and pfp34_v2 source remain unchanged.

The guide covers 55/55 figures, 39 metrics and 26 technique entries. Original questions/hypotheses, expected comparison patterns, observations, interpretation and limitations remain separate. Numerical evidence uses the fresh Archie result, not older handover values. The reviewer supports search/families/tree, related figures, fit/actual/zoom/pan, SVG access, source previews and immediately saved notes/status/resume. Only personal review state is mutable.

Validation: all 25 tests passed (13 original scientific tests plus 12 contract/failure/persistence tests); headless guide/layout/state checks, Python compilation and Slurm shell syntax passed. Exact migration and invariance evidence is in the two validation JSON files. No simulations, bootstrap/CV refits, jobs or active-learning changes were performed.

Limitation: this environment has no DISPLAY or Xvfb. A real launch confirmed a clear no-display error without creating state; rendered GUI usability and host SVG launching still require the ThinLinc environment. Do not treat headless tests as a rendered smoke test or new scientific validation.

Launch in the ThinLinc desktop terminal:

```bash
cd /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/metric_dev
python scripts/review_metric_figures.py --results results/archie_8193723_complete
```

Use `--check` for read-only headless validation. Review README installation guidance before copying over a live legacy result, to avoid leaving old image directories or replacing personal state. The archive sits beside the package and includes the unchanged pfp34_v2 reference material; Python caches are omitted.

---

## Historical v2 handover (retained)

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
