# PFPML metric development and guided figure review

Current supplied result: `results/archie_8193723_complete`. Slurm job 8193723 completed the fresh Archie reproduction. Earlier v1/v2 handovers are retained as historical records; their numerical results and statements about an untested Archie environment do not describe this fresh run.

## Launch the scientific reviewer on Archie

Run in a **ThinLinc desktop terminal**, using the existing Python environment with tkinter and Pillow:

```bash
cd /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/metric_dev
python scripts/review_metric_figures.py --results results/archie_8193723_complete
```

No Slurm job, web server, database or installation step is needed to review the existing results. If the terminal uses another Python environment, activate the existing working annotation-GUI environment first.

Read-only checks (no display needed):

```bash
python scripts/review_metric_figures.py --results results/archie_8193723_complete --check
python scripts/workflow.py validate-figures --output results/archie_8193723_complete
python -m unittest discover -s tests -v
```

The complete delivery archive contains `metric_dev/` and the unchanged reference `pfp34_v2/` material. Stage the archive beside the existing project package and inspect it before replacing files. Do not blindly overlay the archive on an unmigrated figure tree: that would leave the legacy images behind. The durable migration command below can reconcile an existing result before installing the updated manifest/guide. Preserve any existing `figure_review_state.json`; it is deliberately absent from this unused delivery.

## Review controls

- Previous/Next buttons; Left/Right or P/N when not typing in a text field.
- Family dropdown or expandable figure tree; search IDs, titles, families, metric IDs and technique IDs.
- Fit/F; actual pixels/0; +/− zoom; drag to pan; wheel to scroll; Ctrl+wheel to zoom.
- Adjustable image, navigation and explanation panes. The PNG is the review item; SVG is its vector copy.
- Overview explains axes/context; Metrics gives definitions; Design separates original questions/hypotheses from expectations; Read plot gives inspection instructions; Result contains the completed observation and numeric evidence; Limits and Sources remain separate.
- Related-figure dropdown navigates directly. Source preview is read-only and explicitly limited to the first 100 CSV rows/16 columns or 150,000 text characters. Exact paths/filters remain visible.
- Open SVG launches the system viewer and copies the path. Missing handlers show the copied path without terminating the reviewer.
- Review states: Not reviewed, Understood, Revisit, Important. Notes save on every edit, and the last figure resumes on restart. A save failure blocks navigation/close until resolved.

Only `figure_review_state.json` is mutable review content. Writes use a same-directory temporary file, file fsync and atomic replacement (directory fsync where supported); temporary files are cleaned. On Linux a directory advisory lock prevents two reviewer sessions on one result. External state edits are detected rather than overwritten. Filesystem durability depends on the host filesystem. Do not manually edit state while the reviewer is open.

## Package map

| Location | Purpose |
|---|---|
| `scripts/review_metric_figures.py` | Desktop reviewer, guide validation, headless state self-test |
| `scripts/figure_layout.py` | Shared layout contract, hashing, axis validation and safe migration |
| `scripts/plot_metric_analysis.py` | Future PNG/SVG output and logical manifest |
| `scripts/workflow.py` | Explicit analysis/plot/migration/check entrypoints |
| `tests/test_metric_analysis.py` | Existing scientific tests, now checking the actual Archie result |
| `tests/test_figure_review.py` | Layout, guide, persistence and failure-path tests |
| `results/archie_8193723_complete/figures/png/<family>/` | 55 human-browsing PNGs |
| `results/archie_8193723_complete/figures/svg/<family>/` | Corresponding 55 vector copies |
| `plot_manifest.csv` (inside result) | One row per logical figure, explicit paths and SHA256s |
| `figure_guide.json` (inside result) | Authoritative reusable definitions, 55 explanations and provenance |
| `figure_layout_validation.json` (inside result) | Exact pre/post migration hashes and protected scientific hashes |
| `figure_review_validation.json` (inside result) | Final checks and acknowledged display limitation |
| `data/`, `tables/`, `supervised_validation/`, `score_definitions.csv` | Preserved numerical results, unchanged by review work |

## Scientific scope

Canonical APL is projected XY area per membrane molecule per leaflet. ΔAPL is Relax-100-ns minus first-Pull APL. ddAPL is ΔAPL_B−ΔAPL_M. B=POPE75POPG25_W_WF; M=POPC60CHOL40_W_WF. POPC80POPS20 is outside that pair.

The strict cohort contains 156 QC-passing pairs. Persistent morphology uses 80–100 ns medians. No matching APL statistic exists; a different window is never substituted. Physical deviation, PCA variance and visual deformation are not biological efficacy, toxicity or permeability. Four no-interaction systems per membrane limit null calibration. Source tables and `SUMMARY.md` provide the fresh numerical results; historical handovers must not override them.

The guide distinguishes contemporaneous design, method definitions, completed results, historical decisions, general method explanation and explicit inference. Most analyses were exploratory: no directional hypothesis is invented from their outcome. The reference-comparison and redundancy expectations are supported by section 4 of the original task specification. Newly computed medians/extrema in the guide are descriptive summaries of preserved tables, not recomputed models or bootstraps.

## Figure migration and future output

Explicit operation for an unmigrated result (not needed for this delivered result):

```bash
python scripts/workflow.py migrate-figures --output results/archie_8193723_complete
```

The command preflights all manifest pairs and the exact image set, refuses conflicts, moves existing bytes, collapses matching metadata into one record, validates hashes/paths/images/axes, and removes only reconciled empty legacy family directories. Failed validation restores original image locations and manifest. Repeating it on a migrated result performs validation only. It never searches for or migrates other result sets.

The plotting code writes directly to format-first paths and refuses a legacy tree until explicitly migrated. Regeneration is optional and unnecessary for this delivery:

```bash
python scripts/workflow.py plots --output results/archie_8193723_complete
```

Regeneration changes figure bytes and may require refreshing guide metadata/evidence if the scientific results or plot encoding changed. Keep the guide consistent with the manifest. Future result directories are selected explicitly with `--results`; figure counts and families are discovered, not hard-coded. Use the existing guide schema to add metrics, techniques, sources and figure entries, then run `--check`. The GUI does not generate scientific claims automatically.

## Existing analysis operations

These are distinct from viewing and are not needed for this delivered package. Run CPU analysis only in an appropriate allocation. The workflow requires authoritative master/selection/pair/reference inputs; the large master is not included. Source paths and hashes remain in the original manifests.

```bash
python scripts/workflow.py run --project-root /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids --output results/NEW_ANALYSIS --bootstrap 300 --cv-repeats 3 --dry-run
python scripts/workflow.py apply-frozen --input new_unified_metrics.csv --calibration results/archie_8193723_complete --output results/new_scores.csv
```

Remove `--dry-run` only when intentionally running a new analysis. `complete` performs bootstrap/association work; `validate` is the legacy full numerical completion validator and writes its reporting outputs. Neither is the read-only viewer check. Use `validate-figures` and reviewer `--check` for the current task.

## Validation and limitations

25 tests passed, including all 13 existing scientific tests. All 110 image hashes and 59 protected scientific files match their pre-migration bytes. The guide covers 55/55 logical figures, 39 reusable metrics and 26 technique entries. Validation includes unique identities, paired canonical paths, PNG decoding, axis rules, guide references/provenance, temporary state roundtrip, rejected invalid status, failed-write preservation and external-edit protection.

This execution environment has tkinter/Pillow but no DISPLAY or Xvfb. A real launch returned the explicit no-display message without writing review state. Rendered Tk usability, window resizing and operating-system SVG launching remain unverified on ThinLinc; successful headless tests are not claimed as a graphical or scientific validation. Historical contact-sheet evidence referenced by the old generator was not supplied.
