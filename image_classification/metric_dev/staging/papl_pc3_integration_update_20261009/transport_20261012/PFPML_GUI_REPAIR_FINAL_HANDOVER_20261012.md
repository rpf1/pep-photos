# PFPeptoids GUI repair and visual profiles — 12 October 2026

## 1. Outcome and evidence boundary

**A staged implementation for the existing metric_dev Tkinter GUI is complete. No Archie installation or interactive acceptance is claimed.** One bundle supplies corrected existing stage scripts, one profile support module, a guarded patch of the existing GUI, 18 tests, axis/time audits, validation logs and rollback commands. No parallel GUI, result registry, project controller or scientific database is introduced.

| Evidence grade | Finding |
|---|---|
| VERIFIED_ON_ARCHIE | Earlier user terminal evidence reports the 164-figure installation and 156 selected pairs. The supplied collector ran on Archie; it did not run GUI validation. No fresh live connection was available here. |
| OFFLINE_TESTED | Exported GUI `--check`: 164 figures initially. P0/P1/P2 applications: 168 figures, valid image hashes, review-state roundtrip. 18 tests passed. Reverse P2→P1→P0 rollback restored original registry, GUI, review notes and frozen parameters byte-for-byte. |
| STAGED_ONLY | P0 atlas/navigation repair; P1 six display derivatives; P2 profiles, comparison gallery, saved virtual bins, candidate preview/import. |
| BLOCKED | Exact per-system Pull endpoint/Relax-local translation remains unresolved. Historical local-time plots and binned late comparisons retain explicit timing warnings. Selected photograph pixels were not included in the evidence export; identity/path existence is supported by its inventory, not a fresh inspection of those pixels. |
| NOT_DONE | Archie apply, mouse/keyboard GUI acceptance, photograph replacement confirmation, memory/responsiveness measurement with dozens of entries, blinded severity annotation pilot. No controller implementation, APL recomputation, PCA refit, trajectory editing, ML writes, Slurm activity or cleanup. |

`--check` validates registry paths/hashes and state handling; its historical `paths_hashes_axes=passed` label does **not** independently certify numerical axis ranges. Axis checks here additionally inspect renderer output metadata and the relevant plotting logic.

## 2. Read-only preflight and scientific diagnosis

All 925 exported files matched their source-manifest SHA-256. The sole collector error was missing project-root `SCRIPT_INDEX.md`; GUI source, schema-2 registry, review state, existing installers, frozen parameters, source APL script and all selected CSVs were present. Missing catalogue documentation does not block this bounded repair; it remains relevant to a later system-wide inventory.

The registry has 164 figures, 22 families, 164 valid PNG/SVG pairs. GUI SHA-256 is `1a01e3d1a464fade165779791b693ce7a42952fb0035cd01babc94d4b51718d7`.

| Protected source | SHA-256 |
|---|---|
| figure_review_state.json | `394196864123f099ab0e505eee504559487c3d3a2e120f6c001580a6a3065185` |
| data/pca_parameters.json — 12 models | `bf7623fb3a4a01fa0ec0637eebf67dde8f007782d289188e5020c680452b559d` |
| data/frozen_parameters.json | `0aaf543516ba9d1a69f2af75a3f109ae2be379a4e8218144979b33886c38a047` |

Use the exact machine-readable hashes in `validation/offline_source_audit.json` as authoritative when running checks; transcription in prose is secondary.

The selected/QC cohort contains 35 run_0, 52 cycle_01 and 69 cycle_02 pairs: 156 physical pairs, 312 series. Each exported series has 20 Pull bins, one Mixed bin and 20 pure-Relax bins; saved `x_ns` is the bin start. All phase/frame counts reconcile. Each terminal pure-Relax row has `x_ns=200`, nominal edges 200–205 ns and **one frame**. The canonical code includes `time_ns_combined <= 200`, then applies floor-indexed 5 ns bins. Thus a bin whose nominal end is 205 does not prove frames continued to 205. The failed updater incorrectly rejected this endpoint row.

The canonical combined clock adds the actual maximum Pull time to Relax-local time. Bin labels alone do not establish that exact offset. The repair plots **saved combined x_ns**, never subtracts an assumed 100 ns, never recentres the terminal frame at 202.5 ns, and never shades an alleged local 80–100 ns window. It retains 6,240 pure-Relax means, omits Mixed bins, marks one-frame points with red crosses, and uses 0–205 ns display limits and 40–90 Å² y limits. Pure-Relax means span 43.972–71.016 Å². The missing initial pure-Relax portion is a Mixed-bin exclusion, not reconstructed data.

Six new atlases have 35/35, 52/52 and 69/69 B/M panels. Registry arithmetic is **164−2+6=168**. P0 preserves the other 162 figure records. The two retired IDs remain in the review-state JSON and the old images remain in place; a separate verified installation backup/archive is also created. No note-ID migration is performed. Profiles referring to retired IDs show a warning and retain their notes.

## 3. P1 axes and provenance

`validation/axis_audit.csv` covers all 164 originally registered figures; `time_audit.csv` covers all 312 selected CSVs. The old 24-pair atlases are replaced by P0. Six additional comparable absolute projected-APL views receive display derivatives:

1. pristine_papl_b
2. pristine_papl_m
3. pristine_papl_combined
4. endpoint_late
5. binned_frame_reconciliation
6. temporal_selected_trajectories

Original figures, arrays and IDs are preserved. The registry keeps `original_display` paths/hashes/axes, plus source hashes for each derivative. Standardised images use `_axis40_90` filenames inside existing figure family directories. Both absolute APL axes are standardised for endpoint/late scatter comparisons; only APL panels are changed in the mixed APL/geometry figure. Logical figure count remains 168.

Signed ΔAPL, APL slopes, PCA and geometry axes are excluded from rescaling. Out-of-range displayed means are flagged. Historical local-time arrays in the selected-trajectory view are reproduced with explicit unverified-time labels and no late-window shading. The historical binned/frame comparison remains a **saved historical numerical comparison**, not new evidence of aligned windows. Neither view establishes exact local timing.

## 4. P2 profiles and scientific source handling

The existing GUI receives one `Profiles / Comparisons` button and a Toplevel comparison panel. Its original tree, statuses, notes, image viewer and `--check` remain intact. Named profiles live separately in `visual_comparison_profiles.json` under the existing results directory. Opening the panel does not write this state until a user creates/edits a profile.

Profiles have stable UUIDs, titles, purpose, group notes, ordered figure/photo references, item notes, optional versioned queries and timestamps. Images are not copied. Saves use atomic replacement, fsync and optimistic hash guards while the GUI holds its existing results-directory lock. A failed save reports an error, retains unsaved text and blocks closing/switching. Schema failures preserve the existing file.

Features: create, rename, duplicate references, delete with confirmation, search, independent group/item notes, manual up/down ordering/removal, two-image comparison, membrane/run/text filtering, pan/zoom in a full-image window, and missing/changed-source warnings. Only two thumbnails are decoded at a time. Figure review notes and profile interpretations are separate stores.

### Photograph reconciliation

The science handoff's 312 `NEW10__...` photo references have zero exact matches in the exported current photo inventory. All 312 physical system identities join to live annotation records by molecule, run and membrane. Those records supply **322 inventory-present image references**. Photo identities come from cohort/annotation provenance, not guesses from filenames. Paths are constrained to `analysis/Photos`; absolute paths and traversal/symlink escapes are rejected.

Current annotation labels/hashes differ from the science handoff. The catalog retains both evidence versions. Captions distinguish the live label from the historical science label. **An inventory-present current photo must not be treated as the same image/class evidence as a historical NEW10 photo.** Adding a locally available photo snapshots its image hash; a replacement later produces a warning. Pixel equivalence to historical science images was not verified.

The 18 candidate comparisons are offered through `Suggested comparisons`: inspect the hypothesis, caveats and notes, then explicitly create a reference-only profile. This does not assign new classes or validate biological effects.

### Virtual APL BINS

Bins operate on the existing selected `matched_late_pAPL.csv` scalar values: raw `pAPL_late_A2`, signed `delta_pAPL_late_ref_A2`, or its magnitude. The source declares a Relax-local 80–100 ns **saved 1 ns frame mean**; this is distinct from the unresolved combined-clock bin translation. Reference is the saved `pAPL_reference_A2`, not an established stationary null.

Widths are 0.5, 1 or 2 Å², with explicit origin, membrane, run and half-open `[lower,upper)` boundaries. Raw APL cannot pool B/M. Preview shows scalar counts, photograph-reference counts, exact metric values, temporal frame SD and source hash. Temporal SD is not uncertainty of the mean. A saved query preserves its definition/source SHA and is evaluated on reopening; changed sources block evaluation until reviewed. Fixed curated membership remains separate. Adding matching photos creates a deliberate membership snapshot.

Sharing a bin does not establish equivalence; final-frame photographs are not automatically representative of late averaged geometry. No severity-rating/ML ground truth is inferred.

## 5. Delivery layout and staging

One transport ZIP contains one `metric_dev/staging/papl_pc3_integration_update_20261009` payload, tests, this handover and a guarded stage-transfer utility. Production changes use the existing stage, existing `scripts/` and existing result/family directories. `payload/` is the installation source; `validation/` stores offline evidence. Transaction-specific `backup_P1_*`, `backup_P2_*` and P0 backups remain in that same stage. Installer-owned temporary rendering directories are automatically removed. Original project files are never cleaned up or relocated.

The original supplied attachments remain unchanged. The transport extraction under HOME is a distribution copy, not a second project GUI/results root. Keep it until acceptance; no automatic deletion is requested.

## 6. Stepwise Archie commands

**First stage and run read-only preflight. Do not run any `--apply` command until that gate is reviewed and authorised. The specification explicitly requires “Staged only until approval” and “Only with user authorisation”.** Close the GUI before staging/apply because its existing lock protects this result.

### A. Extract the single transport bundle and verify it

```bash
PACKAGE="$HOME/PFPML_GUI_REPAIR_VISUAL_PROFILES_20261012"
unzip "$HOME/PFPML_GUI_REPAIR_VISUAL_PROFILES_20261012.zip" -d "$PACKAGE"
cd "$PACKAGE"
sha256sum -c SHA256SUMS
BASE=/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/metric_dev
RESULTS="$BASE/results/archie_8193723_complete"
STAGE="$BASE/staging/papl_pc3_integration_update_20261009"
python3 "$PACKAGE/stage_bundle.py" --base "$BASE"
python3 "$PACKAGE/stage_bundle.py" --base "$BASE" --write-stage
python3 "$STAGE/audit_gui_repair.py" --base "$BASE" --report-dir "$STAGE/validation/archie_preflight"
python3 "$STAGE/complete_papl_cohort_gui_20261009.py" --base "$BASE"
```

`--write-stage` transfers code only into the existing stage and backs up replaced staged source. It does not install the GUI or modify results. It refuses an unexpected GUI hash or a non-164 registry. If the current project differs, stop and reconcile; do not bypass guards. Send the preflight output and `validation/archie_preflight` report for review.

### B. P0 — after authorisation

```bash
python3 "$STAGE/complete_papl_cohort_gui_20261009.py" --base "$BASE" --apply
python3 "$BASE/scripts/review_metric_figures.py" --results "$RESULTS" --check
python3 "$BASE/scripts/review_metric_figures.py" --results "$RESULTS"
```

Inspect six temporal atlases and folder navigation. Verify 168 figures, correct panel counts, 40–90 axes, combined-time labels, one-frame crosses and no assumed local-window shading. Close the GUI before proceeding. User viewing saves review timestamps normally; the installer itself preserves state bytes.

### C. P1 — separate authorisation

```bash
python3 "$STAGE/standardise_apl_axes.py" --base "$BASE" --output "$STAGE/validation/axis_derivatives"
python3 "$STAGE/install_visual_profiles.py" --base "$BASE" --gate P1
python3 "$STAGE/install_visual_profiles.py" --base "$BASE" --gate P1 --apply
python3 "$BASE/scripts/review_metric_figures.py" --results "$RESULTS" --check
```

Rendering writes only staged derivatives. The separate apply keeps original paths/hashes and preserves figure IDs.

### D. P2 — separate authorisation

```bash
python3 "$STAGE/install_visual_profiles.py" --base "$BASE" --gate P2
python3 "$STAGE/install_visual_profiles.py" --base "$BASE" --gate P2 --apply
python3 "$BASE/scripts/review_metric_figures.py" --results "$RESULTS" --check
PFPML_TEST_BASE="$BASE" python3 "$STAGE/tests/test_gui_repair_profiles.py"
python3 "$BASE/scripts/review_metric_figures.py" --results "$RESULTS"
```

Tests compare protected state against the P0 installation checkpoint; run before making new user notes. If the user has already opened the GUI and changed review timestamps/notes, that preservation test must be interpreted against those authorised user changes, not by resetting the state file.

Required packages are the existing GUI/rendering environment: Tkinter, Pillow, matplotlib, pandas/NumPy and the existing metric_dev support scripts. No dependency upgrade is included.

## 7. GUI walkthrough and acceptance

1. Open `Profiles / Comparisons`; choose `New`; name a comparison.
2. Write the hypothesis, observed comparison, limitations and follow-up in the group-note box; choose `Save notes`.
3. Add the currently displayed figure. Add an existing photograph from the identity-mapped selector. Add a second relevant source. The same source can belong to other profiles independently.
4. Select an item: it and the next visible item appear side by side. Use `Move up/down`, `Remove reference`, `Item note` and `Open full image`. Membrane/run/text filters change the view, not membership/order.
5. Choose `APL BINS`; select metric, width, origin, side, run and bin index. Preview intervals/counts. Save the query separately from adding matching photo references. Inspect SD/reference/time caveats.
6. Use `Suggested comparisons` to preview a candidate before creating it. Current photos and historical science annotations have distinct evidence versions.
7. Close/reopen the GUI. Verify exact profile name, order, group/item notes and query restoration. Duplicate it, edit the copy, then confirm the original is unchanged. Cancel a rename or deletion and check preservation.
8. Exercise missing-reference/replaced-source warnings on a test profile without deleting or altering any original image. Verify folder collapse/expand, search, figure selection and original per-figure notes. Test at least dozens of references for responsiveness.

**No interactive GUI pass is claimed from code compilation or headless tests.** These eight steps are the remaining production acceptance work.

## 8. Rollback

Each P1/P2 apply prints an exact `transaction.json` path. Roll back P2 first, then P1, then P0. Close GUI first. The installer checks that targets still match its installed hashes; it refuses to overwrite later edits. Review/profile notes are retained.

```bash
# Substitute the exact transaction paths printed by your Archie apply.
python3 "$STAGE/install_visual_profiles.py" --base "$BASE" --gate P2 --rollback /exact/backup_P2_TIMESTAMP/transaction.json --apply
python3 "$STAGE/install_visual_profiles.py" --base "$BASE" --gate P1 --rollback /exact/backup_P1_TIMESTAMP/transaction.json --apply
python3 "$STAGE/rollback_p0.py" --base "$BASE"
python3 "$STAGE/rollback_p0.py" --base "$BASE" --apply
python3 "$BASE/scripts/review_metric_figures.py" --results "$RESULTS" --check
```

P0 rollback restores the 164-entry registry and original GUI. Its own generated files move into `stage/rollback_retired_P0`; original images remain in place. It does not relocate scientific sources. Existing profile definitions remain saved if the module is rolled back. Re-application after rollback requires reviewing existing staging backups/output ownership; do not bypass pre-existing-file guards.

## 9. Validation evidence and limits

`validation/offline_final_tests.log` contains 18 passed tests: profile CRUD/restart/order/notes, independent references, conflicts, failed saves, path containment, missing IDs, exact half-open boundaries, query provenance/counts/restart, raw B/M restriction, live photo joins, schema protection, source compilation, folder-selection handling, source endpoint handling, axes, unaffected registry records and 12-model/state preservation.

The three apply logs and `offline_final_gui_check.json` record the 168-figure checks. `offline_rollback_test.log` records successful reverse rollback and 164-figure validation. The reference audit contains original SHA-256 evidence; the derivative manifest records actual arrays/figure paths/hashes for this offline rendering. The atlas preview is a review aid, not a second registered figure.

During development, one reused scratch fixture changed unexpectedly during rendering; its backup-hash guard stopped that attempt. A fresh isolated fixture completed all final gates and rollback. An early photo test incorrectly equated systems with image references; it was corrected to validate 312 unique identities and allow 322 actual references. Only the final passed logs support completion claims.

Remaining uncertainties: exact Pull endpoints; historical local-time alignment; equivalence of current photographs to historical science evidence; interactive Tk behaviour and memory performance on Archie. Named annotations are interpretations, not numerical or biological ground truth.
