# Production morphology handover — 29 September 2026

**NO PRODUCTION JOBS SUBMITTED. Package developed and tested offline; not installed on Archie and not released for production submission.**

## Checkpoints

| Checkpoint | Status | Evidence / limitation |
|---|---|---|
| 0: orientation | Partial | Full handoff, current exported tables and available source snapshots inspected. Live `/users` filesystem, current SCRIPT_INDEX and Git state unavailable. |
| 1: filesystem audit | Partial | Exact eight-entry older snapshot inventory and conservative cleanup manifest created. Full current tree/callers cannot be certified. No cleanup applied. |
| 2: design/freeze | Complete | Scientific, identity, output, QC, scheduling and recovery contracts documented. |
| 3: implementation | Complete offline | Unified CLI; extracted geometry and statistics modules; focused tests. |
| 4: population dry-run | Export-derived complete; live blocked | 395 unique candidates; live GRO/XTC validation unavailable for all 395. |
| 5: job preparation | Draft complete | Immutable 395-row blocked manifest; SHA256; array 0–394%10; expected task inventory; guarded dry-run submission. |
| 6: collect/merge/statistics | Offline tested | End-to-end synthetic run and archived pilot fixtures; 386 historical numerical comparisons pass. Real 30-system / four-system reproduction pending. |
| 7: review/handover | Complete to available evidence | README, decision record, contracts, validation, integration recommendation, fixtures and reports included. |

## Exact population from 29 September exports

- **395 candidates**: 188 POPE75POPG25; 183 POPC60CHOL40; 24 POPC80POPS20.
- **181 bacterial/mammalian pairs (362 systems)** across molecule types; 33 other/unpaired systems.
- **167 primary tripeptoid pairs (334 systems)**. All tripeptoid systems across the three bilayers: 358.
- Runs: run_0 = 140; run_ML_cycle_01 = 107; run_ML_cycle_02 = 148.
- Types: 377 peptoids; 18 peptides. All candidates have nmol = 80.
- Peptoids by length: 3 = 358; 6 = 2; 8 = 15; 9 = 1; 12 = 1.
- Peptides by length: 7 = 2; 8 = 15; 13 = 1.
- **Live-certified production eligibility remains undetermined**, not 395 confirmed usable trajectories. The candidate count is independently derived, not hard-coded.

## Exclusions

The canonical export has 471 rows. Of these, 73 lack qualifying labelled STACKED evidence (including uncertain/skipped/unlabelled, legacy/non-STACKED and concentration-ambiguous records). Three further labelled STACKED rows have no exact simulation match:

| Molecule | Physical run | Bilayer |
|---|---|---|
| Nk-Nfex-Nfex | run_Nk-Nfex-Nfex_rerun_test | POPC60CHOL40_W_WF |
| Nk-Nfex-Nfex | run_Nk-Nfex-Nfex_rerun_test | POPC80POPS20_W_WF |
| Nk-Nfex-Nfex | run_Nk-Nfex-Nfex_rerun_test | POPE75POPG25_W_WF |

All **76 canonical exclusions** and their overlapping reasons are in `manifests/export_population/excluded.csv`. The 52 previously reconciled duplicate/stale identities are already absent from the canonical 471; they are not fresh removals by this workflow. The five unresolved legacy-photo exceptions remain outside the STACKED production selection. Ten ambiguous-concentration records are excluded without label propagation. Live validation may introduce further exclusions.

## Outputs and QC

Production calculates baseline/100 ns endpoint/delta and temporal summaries for midplane height RMS, excess area, RMS mean curvature and local-normal thickness. Upper/lower surface area, excess area, height RMS, curvature RMS and signed/absolute asymmetry are included. Windows begin at 30/50/70/80/90 ns; baseline remains first Pull frame.

QC covers input identity/coverage/counts, Fourier rank/grid resolution, residuals, thickness valid fraction, periodic centring/branch margins, crossings, PO4 assignment changes and local-midplane disagreements, missing values, slope/window sufficiency, asymmetry algebra and descriptive outliers. Invalid metrics are never converted to zero. REVIEW_REQUIRED systems are retained but excluded from default production inference. General folded-surface validity is not proven by these checks.

## Validation

- **22 focused tests passed**: analytical geometry, periodic translation, invalid geometry, exact identities/filters/pairs, 395-row regression, archived synthetic-pilot numerical reproduction, read-only input handling, completion/resume, changed inputs, duplicate/corrupt results, guarded submission, collection/recovery and exact canonical text preservation.
- Archived synthetic-pilot regression covers all stored geometry fields over four systems, with rtol/atol 1e-10. This is not real-trajectory reproduction.
- New synthetic 100 ns GRO/XTC lifecycle passed with baseline and sampled time series; inputs stayed unchanged and no XTC cache files were created beside them.
- **386 historical numerical checks, zero mismatches**, including legacy FDR families. Primary statistics: exactly 334 systems / 167 pairs; POPC80POPS20 does not leak into the primary set.
- Slurm scripts pass `bash -n`; Python sources compile. Ordinary submit invocation only printed the command.
- Real pilot reproduction tolerance is rtol 1e-8 / atol 1e-6 in metric units; this gate has NOT yet been exercised on real trajectories.

## Cleanup and integration

**No live move/delete plan is actionable.** The supplied snapshot manifest has KEEP_ACTIVE and UNKNOWN_REVIEW decisions only. Obtain the live inventory and verify callers before proposing exact targets; any approved removal must use `17082026_cleanup/_cleanup_review/` and a deletion audit. No existing project files were moved, deleted, renamed or overwritten.

Keep morphology standalone for the first production run. Later expose the geometry module through an optional production-controller hook. Do not fold heavy geometry into Photograph.py; do not couple its failures to canonical APL output. See `qc/integration_recommendation.md` for the four-option comparison.

## Not run / remaining gates

No real morphology production or benchmark jobs, Slurm submission, ML training, acquisitions, database sync, canonical APL rewrite, annotation edits, photo regeneration, real-pilot rerun, Git commit/push or cleanup.

Continue in an Archie-connected environment. Confirm live source versions/Git and authoritative export locations; rebuild manifest; run full trajectory validation on an authorised compute allocation. Reproduce the 30-system core and four-system leaflet pilot and review benchmark runtime/RSS before release. The candidate pilot subset contains all 30 existing physical pilot identities. The draft 1 CPU / cluster-default memory / 4 h / concurrency 10 request is provisional. Do not extrapolate real runtime from synthetic tests.

INPUT_CHANGED recovery requires checking scientific provenance against the current canonical simulation analysis before resealing a new manifest. A verified changed input is not permission to silently reuse stale APL values.

Next approval: **separate authorisation for compute benchmark/production submission after the live checks**. Cleanup requires its own exact live manifest approval. No additional permission is needed to inspect sources or continue creating package reports.

## Next commands

After installing this folder at the intended location, use the README's exact audit/manifest/validation/pilot-reproduction commands. The first live read-only audit is:

```bash
cd /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/morph_prod
python scripts/morphology_production.py audit --root .. --out qc/live_audit
```

After successful live validation, prepare the real production plan (not the shipped blocked draft):

```bash
python scripts/morphology_production.py prepare --manifest manifests/live/candidates.csv --validation qc/live_validation/validation.json --out slurm/production
```

After real-pilot comparisons, resource review, a successful release-check and explicit user approval, the exact later production-submission command is:

```bash
python scripts/morphology_production.py submit --plan slurm/production --gate qc/release/production_gate.json --submit
```

**That submission command has not been executed.**
