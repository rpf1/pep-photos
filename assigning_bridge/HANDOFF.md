# Annotation on regenerated photographs — continuation

Priority: enable assigning NEW photographs immediately, without waiting for GLY/HIS/path recovery or STACKED assembly.

## Run
Close the original annotation GUI first. Save/extract `PFPeptoids_assign_new_photos.zip` under the project's analysis directory. It contains `assigning_bridge/`.

```bash
cd /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis
python assigning_bridge/activate_new_photos.py
bash image_classification/pfp34_new_photos/START_ANNOTATING.sh
```

Run preparation once. Thereafter use START_ANNOTATING.sh only. Do not continue labelling in the original session: the two sessions are not synchronized.

## Result
- Separate continuation: analysis/image_classification/pfp34_new_photos/.
- `images/`: 412 new composites expected from the supplied report, with filename prefix NEW10; remaining systems retain copied original evidence with LEGACY prefix. Counts reflect live saved session membership, not an assumed hard-coded 523.
- `session/`: cloned SQLite state retaining current labels, notes, status, order, decision timestamps, event IDs and all historical event rows. Evidence fields change to match the new images, recorded in evidence_transition.json.
- `original_session_snapshot/`: original SQLite backup, exports, source/config copied before migration. Original Photos and original session remain in place and unmodified.
- `READY.json`: only written after export, source checks, backup, and reopening tests pass.
- `image_checklist.csv`: NEW10/LEGACY by system, original failure reason, and already-labelled new-image cases for review. It does not clear or alter any label.
- `unresolved_audit.json`: fixed-depth Boxes inventory, same-molecule/bilayer candidate paths, exact-run flags, molecule counts matched against old filenames, and available local GLY/HIS topology lines. Candidates are not automatically substituted, classified or rendered.
- `cleanup_plan.json`: exact candidates that can be archived after GUI verification.

The viewer still enforces image-content hashes. It uses the normal independent system keys. History before the evidence transition is retained byte-for-byte in the events table. Undo is limited to new decisions after the transition, preventing restoration of records tied to old image hashes; earlier labels can still be edited normally.

The preparation holds the original application's session lock to ensure a closed, consistent source. It verifies existing source image hashes against stored evidence and the original CSV exports against the cloned DB. Missing/changed evidence aborts preparation; it does not guess or reset state. A partial unsuccessful preparation is retained for diagnosis and is not launched.

No change is made to canonical Photograph scripts, APL data, active-learning databases, source trajectories or GRO files. The continuation annotator is copied from the user's current installed script, with exactly one guarded edit to its undo query. All six current configured classes are carried through from the live configuration.

## Validation performed locally
Synthetic two-system fixture, using the supplied real annotation tool:
- preserved every non-evidence annotation field;
- preserved historical event rows;
- copied one new ten-panel composite and one legacy image;
- original source hashes unchanged;
- normal evidence checks pass;
- migrated session reopens normally;
- historical undo is blocked at boundary, new decision/undo works;
- new decision remains after close/reopen.
Real Archie migration and interactive GUI usability still require execution. No claim is made that all beads are visible in every rendered production image.

## Cleanup
After confirming the new GUI displays NEW10 images and saves/reopens correctly:
```bash
python assigning_bridge/activate_new_photos.py --cleanup-confirmed
```
This verifies every planned candidate hash and archives only the downloaded patch installer and completed, recognizable pilot directories into:
`17082026_cleanup/_cleanup_review/photograph_installer_<timestamp>/`.
The archive contains deletion_audit.csv recording moves, not deletions. Unknown files inside a pilot directory exclude that entire directory from cleanup. Incomplete pilots are retained. No arbitrary project-wide cleanup is attempted.

Protected: original Photos, pfp34_v2, new continuation, bulk regeneration tree, patch source, unresolved inputs and all scientific files. These are still needed for provenance, recovery or further work. Full removal/consolidation of older sessions/images is deferred until user acceptance of the continuation; never silently delete them.

## Open work
Resolve 26 GLY/HIS systems, 73 missing paths and 10 multiple-count cases; 2 Nkeqm cases explicitly deferred. Existing fallback images may contain omitted residues, so the LEGACY marker is scientifically significant. Assemble full multi-bilayer STACKED images later; the annotation tool needs direct per-system composites, so assigning need not wait.
