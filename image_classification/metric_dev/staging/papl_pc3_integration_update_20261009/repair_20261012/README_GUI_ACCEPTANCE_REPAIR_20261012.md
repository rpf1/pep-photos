# PFPeptoids — targeted GUI acceptance repair, 12 October 2026

**Scope:** patch the **existing** `metric_dev` Tkinter comparison module and optionally register six **secondary** Pull/Mixed/Relax pAPL atlases. Do **not** rerun earlier P0/P1/P2 installs. No MD, APL, PCA, ML, photograph generation or scientific recomputation. This package does not overwrite saved `visual_comparison_profiles.json`, review notes, reference photographs, canonical source CSVs or the six primary Relax-only atlases.

## Completed offline validation

- The supplied 12 October source archive contains the entire 156-selected-pair / 312-system binned-APL cohort. Read-only row reconciliation succeeded: **12,792 rows = 6,240 Pull + 312 Mixed + 6,240 Relax**. Each of the 312 source files has 41 phase-labelled bins. All 312 terminal 200 ns points contain one frame; all 312 nominal 100–105 ns mixed bins contain frames from both phases. There were **zero** values outside the requested 40–90 Å² plotting range in this archive.
- **Six full-phase optional atlas pairs of PNG/SVG files were rendered successfully** from those archived real source CSVs; per-image hashes are in `rendered/render_report.json`. These files are not claimed to be live Archie output. The installer **must check source snapshot hashes against the live Archie files before installation**; it stops on any mismatch.
- **Ten new deterministic regression tests pass** for suggested-comparison identity, duplicate prevention, legacy-profile recognition, note preservation, atomic failure, registered pristine-figure lookup, and independent duplication. Earlier user-reported 18/18 baseline tests and GUI interactive acceptance predate this patch. **No live-Archie patched-GUI mouse/keyboard acceptance is claimed.**

## Corrected GUI behaviour

1. Suggested comparisons: selection is read-only; **Open existing profile** reopens the same profile UUID and does not re-import or overwrite notes. **Create profile from suggestion** is explicit and writes one complete profile atomically; a second create for the same candidate is refused, with instructions to open existing. A **Duplicate** action in the existing profile window remains available for deliberate independent copies. Multiple existing matches are listed for explicit user selection; never merged/deleted. Candidate identifiers are based on immutable pair IDs, membrane side and evidence reference; not the mutable display title. Pre-patch imported profiles are matched conservatively against original pair identities and scientific purpose. If a legacy profile cannot be authenticated, do not auto-link it: inspect it manually before creating another. This is protection against accidental duplicate interpretations, not a destructive migration.
2. **Add photograph** opens a clearly labelled common source selector: **Scientific figures** and **Membrane photographs** are distinct search categories. `pristine` matches *registered pAPL figure IDs/titles/paths* if present. The 312 mapped peptoid photographs remain in their own category. No missing pristine final-frame photographs are claimed or fabricated.
3. The six pre-existing Relax-only atlas IDs and pixels stay unchanged. The six **optional** secondary figures have `papl_full_pull_relax_...` identifiers and reside in the existing `temporal` family. Grey=Pull, blue=Relax, orange diamond=Mixed, red cross=one-frame point; segments are **not connected across Mixed**. The nominal 100–105 ns Mixed bin is **not claimed as an exact physical stage boundary**. Initial equilibration is **not silently discarded** because there is no independently authenticated per-system exclusion interval. Recorded phase labels, combined time, immutable saved pAPL and 40–90 Å² limits are used. Primary figures are unchanged.

## A. Transfer and stage (copy ZIP to your Archie HOME first)

Close the running GUI. All files are staged **only** in the existing project stage:

```bash
BASE=/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/metric_dev
RESULTS="$BASE/results/archie_8193723_complete"
STAGE="$BASE/staging/papl_pc3_integration_update_20261009"
FIX="$STAGE/repair_20261012"

unzip -n "$HOME/PFPML_GUI_ACCEPTANCE_PATCH_20261012.zip" -d "$STAGE"
cd "$FIX"
sha256sum -c SHA256SUMS
python3 "$FIX/tests/test_acceptance_repair_20261012.py"
```

`unzip -n` avoids overwriting any already-staged fix. Do not extract the ZIP into canonical `scripts/` or `results/`.

## B. Read-only preflight: both parts

```bash
python3 "$FIX/install_gui_acceptance_fix.py" --base "$BASE"
python3 "$FIX/optional_full_phase_atlases.py" --base "$BASE"
```

The first command **requires the exact shipped P2 profile-module SHA-256** `235d994b8fe6682daf9b26097740ef02dbb4a08efe5d87f3a79785b73636b213` and the installed 168 (or 174) registry. Stop and investigate a mismatch; do not bypass the guard. The second command recomputes only read-only source checksums/row interpretations. If it reports **prepared six-atlas source fidelity: MATCH**, the pre-rendered images can be installed without plotting work. If it reports **MISMATCH**, use the explicitly guarded `--rerender` option in Section D; it deletes **only verified, package-owned staged render files**, never canonical results or saved state.

## C. Apply targeted GUI fix and validate

```bash
python3 "$FIX/install_gui_acceptance_fix.py" --base "$BASE" --apply
cd "$RESULTS"
python3 "$BASE/scripts/review_metric_figures.py" --results "$RESULTS" --check
python3 "$BASE/scripts/review_metric_figures.py" --results "$RESULTS"
```

The installer saves the exact prior module and transaction under `$FIX/backups/gui_acceptance_*/transaction.json` and restores it automatically if headless GUI validation or protected-state hashes fail. Interactive test: create a suggested profile, annotate it, close/reopen, choose **Open existing profile** from the suggestion, verify the **same UUID and notes** and no new profile; then type `pristine` in **Add photograph**, where the entries must explicitly display `[FIGURE — scientific plot]` rather than fabricated photos.

## D. Optional full-phase figures; install only after source-fidelity pass

If Section B says `MATCH`, **skip rendering**:

```bash
python3 "$FIX/optional_full_phase_atlases.py" --base "$BASE" --apply
cd "$RESULTS"
python3 "$BASE/scripts/review_metric_figures.py" --results "$RESULTS" --check
```

If the prepared source hashes **do not match** live Archie files, do **not** apply old figures. Re-render only the preserved existing 5 ns outputs, no source science:

```bash
python3 "$FIX/optional_full_phase_atlases.py" --base "$BASE" --rerender
python3 "$FIX/optional_full_phase_atlases.py" --base "$BASE"
# Require MATCH before this final command:
python3 "$FIX/optional_full_phase_atlases.py" --base "$BASE" --apply
```

Rendering is resumable if interrupted: each completed PNG/SVG pair is verified before skipping; the source fingerprint prevents reuse against changed CSVs. This renderer may take substantial CPU time for the largest 69-panel atlases. The apply transaction stores original registry bytes, hashes source assets and protects profile/review/PCA state. GUI `--check` must validate all 174 figures or the installer automatically rolls back just this optional addition. The six primary pure-Relax atlases are retained intact.

## E. Rollback just this targeted patch, if necessary

Close the GUI and substitute the actual transaction filenames printed by `--apply`:

```bash
# If both parts were installed, roll back the optional six figures first:
python3 "$FIX/optional_full_phase_atlases.py" --base "$BASE" --rollback "$FIX/backups/full_phase_TIMESTAMP/transaction.json"

# Roll back the profile-module fix without touching saved profiles:
python3 "$FIX/install_gui_acceptance_fix.py" --base "$BASE" --rollback "$FIX/backups/gui_acceptance_TIMESTAMP/transaction.json" --apply
```

Both rollback paths hash-guard current targets and refuse overwriting a later external change. Do not roll back P0, P1 or P2. Never delete the saved user profile JSON or the earlier P1/P2 transaction backups.

## F. Scientific/implementation restrictions and honest gates

- No source imaging in the prepared archive proves that a separate pristine **membrane photograph** exists; registering a pristine scientific plot is not the same thing.
- The `x_ns` saved column is the combined 5 ns **bin start**, not the bin centre. The terminal bin nominally 200–205 ns contains one frame at 200 ns. Do not infer a 205 ns measured endpoint.
- The mixed bin's existence only bounds the nominal transition region; it does not establish the exact maximum Pull time. Phase labels derive from the canonical CSV.
- **Do not discard a first 5 ns bin as 'equilibration' without source evidence.** Binned intervals are not equilibration annotations.
- The six optional plots are descriptive comparisons, not biological efficacy or replicate confidence evidence.
- If frozen figures are now 174, no other script may assume 168 after this optional *additive* installation.
- Verify actual GUI behaviour on Archie. The tests here are source-level and headless; no live GUI intervention was performed from this conversation.

## Transport contents and cleanup

`visual_comparison_profiles.py` is the complete replacement profile module based on the shipped P2 payload. `install_gui_acceptance_fix.py` installs only that module. `optional_full_phase_atlases.py` installs/rolls back only its six new registered figures, one derived source-manifest snapshot and the changed figure registry. `rendered/` holds precomputed *staged* images/source evidence; `tests/` contains ten new tests. The rest of `metric_dev` and all earlier stages remain untouched.

After manual acceptance and after ensuring rollback transactions are retained, the transport ZIP in HOME can be removed if desired. Keep `$FIX` while the change remains installed because it contains the source, evidence and rollback transactions. No disposable files are added to unrelated canonical project directories.
