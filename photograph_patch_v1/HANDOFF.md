# PFPeptoids photograph repair — staged pilot

## Status
Implementation prepared in separate copies. Local contract checks pass. **Real VMD rendering and visual validation have NOT run.** No production installation, regeneration, annotation migration or deletion performed. All nine files extracted from the supplied ZIP remain byte-identical. No GRO/XTC files or VMD executable were supplied.

## Files changed
- Photograph.py: shared registry import and GRO preflight; nine snapshot order/canvases; absolute input paths; explicit final trajectory frame via template; rendering inside new output directories; raw snapshots retained; VMD failures checked; default bulk scan blocked for pilot.
- Photograph.tcl: shared residue registry, unknown-residue guard and nonempty-polymer guard, independent top/X/Y orientation restoration, nine snapshots.
- Photograph_components.py: same validation and registry, inherited nine-view layout; requires fresh diagnostic output directory.
- Photograph_components.tcl: shared registry and runtime guard; same independent nine-view orientations. Component-specific rendering style retained.
- New photograph_residues.tcl: canonical peptide/peptoid name groups, parsed by Python and sourced by Tcl.
- New photograph_validation.py: strict fixed-width GRO names and explicit environmental exclusions; no directory-name substitution; all nonenvironment residues must be known.
- New validate_photographs.py and run_pilot.sh: explicit targets only, default preflight, optional isolated render.
- New test_photographs.py, local_validation.txt, source_hashes.json and changes.diff: validation and review evidence.

## Mapping decisions
| Actual/legacy names | Category | Existing colour |
|---|---|---|
| Na, Nx | Hydrophobic | ColorID 3 |
| Nfnap, Nfnaph | Aromatic | ColorID 11 |
| NfeC4, Nfe4Cl | Aromatic | ColorID 11 |
| NfeB4, Nfe4Br | Aromatic | ColorID 11 |
| Nkeqm | Unsupported, deliberately deferred by user | No rendering; explicit failure |

No GRO names are changed. Existing standard peptide/peptoid groups are retained. The old Python-only bracket spellings were not actual GRO names and are replaced by the shared Tcl aliases. Unsupported environmental residues also fail closed; review their identity rather than automatically treating them as peptoids.

## Layout and camera
Fixed order: top+peptoid | top | side-X+peptoid | side-X | side-Y+peptoid | side-Y | water-X | water-Y | bottom.

Original filenames are retained for compatible old panels: snapshot_top.png contains polymer; snapshot_top_bilayer.png does not. Added snapshot_side_y.png, snapshot_side_y_bilayer.png and snapshot_water_y.png. Top rotation matrix is saved once and restored before every view. X=-90 degrees; Y=-90 degrees; bottom X=-180 degrees. No X-to-Y accumulated rotation. Side-X means rotation about X, not viewing along X.

Original colours, representation parameters, trimming, fonts, title rules and panel scaling remain. Side-Y canvas matches side-X (330x250); water-Y matches water-X (330x330). Composite width 2970 px; height 372 px without system title, plus existing variable title band. The basic template now explicitly selects the final loaded frame, matching components; confirm this against old renders before claiming exact visual equivalence.

## Run on Archie
Extract this package into a NEW directory:
`analysis/photograph_patch_v1/`.
Do not replace the canonical analysis scripts yet.

From that directory:
```bash
bash run_pilot.sh
bash run_pilot.sh --render
```
First command checks actual Relax.gro coverage and Relax.xtc availability without rendering or creating output. Second renders into a timestamped `analysis/photograph_validation/pilot_*` directory. Needs the same MDAnalysis, Pillow, VMD/vglrun, ImageMagick and graphical environment as the existing pipeline. It may open VMD windows. No Slurm jobs are submitted.

Explicit pilot targets:
1. Nfnap-NfeC4-NfeC4 / POPE75POPG25_W_WF / run_0.
2. Na-Nab-Ni / POPC60CHOL40_W_WF / run_0 (audit confirms actual Nx in Relax.gro).
3. Nwe-NfeB4-Nke / POPE75POPG25_W_WF / run_0.
4. Nk-Nm-Nke / POPE75POPG25_W_WF / run_ML_cycle_02 (ordinary control; actual directory availability still checked on Archie).
All four identities appear in the supplied annotation CSV. The pilot uses the conventional _80 simulation directory names and stops if inputs are missing. Nkeqm is covered by a rejection fixture, not a rendered case, per revised scope.

Return console output, validation.json and the four horizontal composites for review. Check all beads visually, X/Y orientation, order, crop and camera scale. Components renderer has local orientation tests but still needs its own actual VMD visual test if it will be used in production.

## Completed local tests
- Synthetic fixed-width GROs: all four mapping cases covered, expected polymer atom counts preserved.
- Unknown Nzz, deferred Nkeqm, and solvent-only GROs rejected.
- Both actual Tcl view procedures executed using Tcl with mocked VMD commands and numerical rotation matrices: nine expected filenames, orthogonal side directions, restored top, matching X/Y water orientations.
- Real Pillow compositor produces nine columns at 2970x372 without title.
- Python compilation passes.
- Uploaded source files byte-identical after work.
These tests do not establish actual VMD bead visibility, GPU rendering, trajectory correctness or source-tree immutability on Archie. The pilot checks input GRO hashes; a full before/after filesystem audit is still required for production acceptance.

## Annotation constraint — unresolved production gate
Current annotate.py source_problem recomputes evidence_sha256 from image membership/content. Replacing/adding photographs blocks decisions while retaining labels. There is no safe existing automatic evidence-rebaseline command. Do not reset SQLite, recreate manifests, initialize a new production session or disable the guard. No annotation source/config/CSV/database is changed by this package.

Before production replacement, design and review an explicit provenance-aware evidence transition preserving system keys, labels, status, notes, timestamps, order and event history, recording old/new image hashes and requiring review of previously labelled affected systems. This is a separate gate; current authorization prohibits annotation-state modification.

## Proposed regeneration scope and provenance procedure — NOT EXECUTED
Use the current annotation dataset's exact molecule/bilayer/run keys, not all 769 audited directories. Refresh that dataset snapshot before deciding scope. Exclude/defer Nkeqm explicitly. Identify affected GROs from actual names and compare registry coverage; render complete nine views for affected systems. For unaffected systems reuse old raw six panels only when input frame, rendering settings, crop and provenance match; render Y-only views where feasible. Current package does not implement selective Y-only regeneration: pilot rerenders all nine for comparison. The original basic pipeline deletes raw images, so determine actual raw availability first; do not cut images from labelled composites and call them equivalent raw evidence.

After pilot approval: snapshot/back up complete annotation session while closed, inventory/hash existing photos, generate replacements outside Photos, review, then propose exact archive destinations and atomic replacement operations. Preserve old photographs under the established project cleanup/provenance scheme with an audit mapping; never delete first. Execute neither bulk regeneration nor evidence migration until that concrete plan is approved.
