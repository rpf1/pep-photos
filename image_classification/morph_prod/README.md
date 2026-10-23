# Production morphology

**Offline implementation validated; live production release is blocked. NO PRODUCTION JOBS SUBMITTED.**

Install this whole folder, without overlaying an existing package, at:
`/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/morph_prod/`.
This build was created in an isolated workspace because the live Archie filesystem was unavailable.
Read `HANDOVER.md` for completed checkpoints and remaining gates.

## Entry point and layout

`python scripts/morphology_production.py --help` is the only workflow entry point.
`morphology_geometry.py` holds the extracted numerical kernel; `morphology_statistics.py` owns inferential calculations.
`config/default.json` centralises settings. `tests/fixtures/` contains clearly identified synthetic trajectories and historical regression tables, never production observations.
`manifests/export_population/` is the 395-row export-derived candidate list.
`slurm/production_draft/` is an immutable, **blocked** 395-task review plan, not a validated live manifest.
`qc/` contains checks and the partial filesystem audit. `results/` is empty.
Generated products stay outside source directories. Commands refuse existing output directories; use a new short run identifier for a real retry.

## Scientific contract

- Periodic Fourier graph surfaces, PO4 geometry markers, wavelength cutoff 4 nm, 32 × 32 grid; orthorhombic cells only.
- Height RMS: projected-grid RMS about each surface's own mean, nm. Core system metric uses the midplane.
- Excess area: surface area / box XY area − 1, dimensionless fraction. Core system metric uses the midplane; upper/lower are separate.
- Mean-curvature RMS: area-weighted RMS of graph mean curvature, nm⁻¹. Same +z graph convention on both leaflets; RMS is sign independent.
- Local-normal thickness: midplane-normal intersections with both PO4 surfaces; median weighted by midplane area Jacobian, nm. It is a bilayer thickness. No invented upper/lower half-thickness.
- Baseline is the **first Pull trajectory frame**, not Pull.gro coordinates and not a membrane-only equilibrium. Endpoint is Relax 100 ns within 0.051 ns.
- Sample every tenth Relax frame and include the endpoint. Retain median/P05/P95, slope per ns, valid fraction and descriptive fraction above baseline for 30–100, 50–100, 70–100, 80–100 and 90–100 ns. At least three sampled frames per window.
- Preserve upper/lower areas, excess area, height RMS, curvature RMS; signed upper − lower and absolute differences. Delta always means endpoint − baseline, including delta of absolute asymmetry.
- Canonical BOX APL and ddAPL are never replaced. The existing source `thickness_change` has its own units/sign (typically Å and start − finish); it is not silently converted to local-normal thickness change.
- No production CHOL assignments, local enrichment, flip-flop inference, composite target or ML training.

## Identity and eligibility

Use the current **canonical annotation export plus canonical simulation CSV**, not logical ML acquisitions.
Physical key is `molecule|integer_nmol|bilayer|exact_run`, parsed from the actual simulation path.
Every supporting image must resolve exactly the same nmol. Multiple images are one observation.
Only canonical, labelled STACKED rows with a unique exact simulation, valid metadata and recorded 100 ns completion become candidates.
Unknown/mixed concentrations and unsupported/missing exact matches are excluded with reasons.
The known ADA correction comes from `authoritative_run`; no new alias heuristic is introduced.
POPC80POPS20 is a separate role. Its observed metadata is 510 PO4 / 510 membrane residues; the code validates source-derived counts instead of assuming 512.

## Typical commands

Run the following from the installed package, using a Python environment containing `config/requirements.txt`.
These commands create new package files only. The two source paths are the locations documented by the 29 September export and must be confirmed in the live audit.

```bash
cd /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/morph_prod
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MORPH_PYTHON="$(command -v python)"
python scripts/morphology_production.py audit --root .. --out qc/live_audit
python scripts/morphology_production.py manifest --annotations ../pfp34_new_photos/session/canonical_export_20260929/canonical_annotations.csv --master ../../masterdataanalysis.csv --out manifests/live
python scripts/morphology_production.py validate --manifest manifests/live/candidates.csv --out qc/live_paths
```

The lightweight validate command intentionally exits 2 until trajectory checks pass. Run the full check on an **already authorised compute allocation**, not the login node:

```bash
python scripts/morphology_production.py validate --manifest manifests/live/candidates.csv --read-trajectories --out qc/live_validation
python scripts/morphology_production.py prepare --manifest manifests/live/candidates.csv --validation qc/live_validation/validation.json --out slurm/production
```

Do not overwrite `slurm/production_draft`. Preparation never submits.
Filters are available on `manifest`: `--tripeptoids-only`, `--paired-only`, repeatable `--bilayer`, `--run`, `--molecule`, or `--manifest` to filter an existing physical manifest.
Pairing requires the same molecule, nmol and physical run, and exactly one bacterial and one mammalian system.

## Pilot reproduction and benchmark before release

Use the 30 physical identities in `/users/pfb19164/pfp_morphology_pilot/results/balanced_panel_systems.csv` to select rows from the live candidate manifest. The prepared export-based selection is `manifests/pilot_benchmark/candidates.csv`; rebuild it if identities change.
Validate that selection on a compute allocation and `prepare ... --benchmark` into `slurm/pilot_benchmark`.
A later **separately authorised** benchmark submission uses `submit --plan slurm/pilot_benchmark --benchmark-only --submit`; this package has not executed it.
Thirty pilot systems are for scientific reproduction; inspect runtime/RSS on the first four representative systems before scaling. Prefer one CPU, BLAS threads fixed to one. The draft uses 1 CPU, no explicit Slurm memory request, a 4 h time limit and at most 10 simultaneous tasks. Memory is left to the cluster default unless measured evidence later justifies an explicit request. The current per-system implementation is single-process and BLAS/OpenMP threads are fixed to one, so requesting 4 CPUs would not accelerate it without a code change. Benchmark wall time and RSS on representative 1-CPU tasks before the final production plan.

After benchmark completion, collect and compare actual results:

```bash
python scripts/morphology_production.py collect --plan slurm/pilot_benchmark --out results/pilot_reproduction
python scripts/morphology_production.py verify-pilot --reference /users/pfb19164/pfp_morphology_pilot/results/morphology_v1/grid32_stride10_lambda4nm/morphology_v1_summary.csv --reference-manifest /users/pfb19164/pfp_morphology_pilot/results/balanced_panel_systems.csv --candidate results/pilot_reproduction/system_morphology.csv --out qc/pilot_reproduction
python scripts/morphology_production.py verify-pilot --reference /users/pfb19164/pfp_morphology_pilot/results/morphology_leaflets/smoke/summary.csv --reference-manifest /users/pfb19164/pfp_morphology_pilot/results/balanced_panel_systems.csv --candidate results/pilot_reproduction/system_morphology.csv --leaflets --out qc/leaflet_reproduction
python scripts/morphology_production.py release-check --pilot-check qc/pilot_reproduction/pilot_verification.json --leaflet-check qc/leaflet_reproduction/pilot_verification.json --statistics-check qc/statistics_regression/statistics_verification.json --audit qc/live_audit/orientation.json --live-pilot-source /users/pfb19164/pfp_morphology_pilot/scripts/06_morphology_v1.py --benchmark-plan slurm/pilot_benchmark --out qc/release
```

Pilot comparisons require the same physical run/concentration, not just molecule names. Real-data comparison tolerance is rtol 1e-8, atol 1e-6 in the reported metric unit. Any mismatch blocks release; do not widen tolerances to force agreement.
Release also checks the live geometry functions against the frozen extraction. Changed live source requires inspection and fresh regression, not an override.

## Later production submission — not performed

Only after live gates pass and the user approves:

```bash
python scripts/morphology_production.py submit --plan slurm/production --gate qc/release/production_gate.json --submit
```

Omit `--submit` for print-only dry-run. Never call bare sbatch as a shortcut around the gate.
The array maps index 0…N−1 to exactly one immutable row and sets Python/thread environment explicitly.
Each task records input SHA256/size/mtime, manifest hash, code/config hashes, identity, host, Python/dependency versions, start/end times and outcome.
Input hashes are strong, but reading all trajectories for hashing/coverage has I/O cost; do this on compute nodes.
The XTC reader suppresses offset/lock-file creation beside protected simulation inputs; MDAnalysis 2.10.0 is pinned and tested.

## Completion, recovery, merge and statistics

```bash
python scripts/morphology_production.py status --plan slurm/production --out qc/production_status
python scripts/morphology_production.py recover --plan slurm/production --out manifests/recovery
python scripts/morphology_production.py collect --plan slurm/production --out results/production_collection
python scripts/morphology_production.py merge --master ../pfp34_new_photos/session/master_analysis_20260929_v1/01_system_master_extended.csv --morphology results/production_collection/system_morphology.csv --out results/production_master
python scripts/morphology_production.py stats --master results/production_master/master_with_morphology.csv --out results/production_statistics
```

Recovery produces a new manifest only; validate and prepare it in a new plan directory. COMPLETE outputs are not recomputed. RUNNING/UNKNOWN is deferred; explicitly supply `--confirmed-stopped-task ID` only after checking the scheduler. Old partial directories are preserved. `collect --recovery-plan PATH` can include a completed recovery plan and rejects competing completed results. An unresolved original RUNNING record remains blocked until its stop is recorded/reviewed; do not alter protected simulations to resolve this.
Collector validates completion hashes, all required metrics/QC, identities and input provenance. Partial collection is labelled PARTIAL, with an explicit exclusions table. Raw frames stay in task directories.
Merge is one-to-one and preserves **every existing column value as text**, including canonical APL and annotation provenance. New metrics use `production_morph__`; existing pilot columns remain `morph__`.
Primary statistics retain clean paired tripeptoids and require PASS morphology for both members. Secondary systems are separate. Paired contrasts use Wilcoxon; per-membrane class tests use Kruskal–Wallis/Mann–Whitney; correlations use Spearman; BH corrections have declared output families. Label-pattern ddAPL tests require at least five pairs per pattern. Rare morphology classes (<3 systems) remain descriptive. Repeated sequence observations in one membrane stratum block inference until a clustered model is specified. Leaflets are never replicates. No new ML target is selected.

## Tests

```bash
python -m unittest discover -s tests -v
python scripts/morphology_production.py stats --master tests/fixtures/historical/system_master.csv --pilot --out qc/statistics_recheck
python scripts/morphology_production.py verify-statistics --reference tests/fixtures/historical/statistics_reference --candidate qc/statistics_recheck --out qc/statistics_recheck_validation
```

The second/third commands reproduce historical numbers only; they do not claim expanded production science.
