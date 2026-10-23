# Review corrections

Before Archie installation, the review package was corrected to avoid collision/confusion with the project's existing `workflow.py`.

Active script names are now:

- `scripts/morphology_production.py` — production morphology CLI
- `scripts/morphology_geometry.py` — numerical geometry kernel
- `scripts/morphology_statistics.py` — post-production statistics
- `tests/test_morphology_production.py` — focused tests

The Slurm template no longer requests memory explicitly. Cluster-default memory is used unless a later measured requirement justifies an explicit request.

The current per-system implementation is single-process and fixes BLAS/OpenMP/MKL threads to one. Therefore the initial benchmark should measure wall time and peak RSS on representative **1-CPU** tasks; a 1-vs-4 CPU benchmark would not be meaningful unless the implementation is first parallelised.

All live validation/release hashes must be regenerated after installation; no pre-review hash is a production release gate.
