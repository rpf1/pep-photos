# Integration recommendation

**Keep the standalone CLI for this production run. Later expose its geometry module as a callable component, with an optional scheduling hook from the existing controller. Do not merge now.**

| Option | Assessment |
|---|---|
| A. Standalone | Best current fit: independent array scheduling, retries, versioning, tests, manifest ownership and failure isolation. Extra command is offset by one unified lifecycle CLI. |
| B. Fold into APLanalysis.py | Could share input discovery eventually. The inspected `aplanalysis.py` snapshot owns cumulative APL and many derived outputs. Coupling a heavier reconstruction and QC failure path to canonical APL increases risk and makes partial success harder to explain. |
| C. Fold into Photograph.py | Poor owner for scientific production calculation. Inspected source invokes VMD/Tcl and composes image panels. Rendering choices, bead visibility and morphology numerical validity should have separate failure and provenance records. |
| D. Callable geometry module plus existing entrypoint hook | Preferred future integration: share the validated calculation module, retain standalone job/task lifecycle, let a production entrypoint prepare or schedule an independent morphology stage only when requested. Avoid duplicate Fourier implementations. |

This recommendation is based on copied sources: `original/analysis/Photograph.py` and `evidence/files/analysis/aplanalysis.py`. Neither is certified as the current live source. The available `SCRIPT_INDEX.md` is an earlier consolidation proposal, not the current project-root catalogue. No source was modified, committed or pushed. Live caller/import and nested-repository inspection remains necessary before any integration patch.

Future ML use should consume a versioned, QC-filtered derived table joined by physical identity. It should not query photograph state or reinterpret canonical ΔAPL. No composite score or target change is recommended before the expanded analysis.
