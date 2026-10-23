# Flatness ↔ physical-response Slurm analysis

Purpose: rapidly compare manual membrane phenotype labels against quantitative morphology, ΔAPL/ddAPL, matched bacterial/mammalian responses, metric redundancy, discordant systems, and temporal persistence.

## Parallel tasks

0. Strict morphology ↔ master APL join, annotation/QC cross-check
1. Manual phenotype ↔ quantitative morphology
2. Manual phenotype ↔ ΔAPL, overall and per membrane
3. Primary matched tripeptoid ddAPL ↔ bacterial–mammalian structural differences
4. Discordant cases (small ddAPL / large structural difference and converse)
5. Metric redundancy / Spearman correlations
6. Manual-label membrane association + endpoint-vs-late-window persistence

All primary inference uses `morph__qc_status == PASS`.
The strict master join is concentration-safe (`nmol == 80`) and does not force unmatched rows.

Two binary phenotype sensitivity definitions are used:
- conservative: curved = `curved`; flat = `flat_interaction` + `flat_no_interaction`
- inclusive: curved = `curved` + `bulge_curved`; flat = `flat_interaction` + `flat_no_interaction` + `bulge_flat`
- `pore_formed` is excluded from binary tests but retained in original-class summaries.

`POPC80POPS20` is never used in the primary bacterial–mammalian ddAPL pair analysis.

## Run on Archie

Place these four files together, then:

```bash
chmod +x submit_flatness_physics.sh
bash submit_flatness_physics.sh
```

The launcher auto-discovers:
- `/analysis/masterdataanalysis.csv`
- the current production `system_morphology.csv`
- the 29 September canonical annotation export

If auto-discovery fails, pass exact files explicitly:

```bash
bash submit_flatness_physics.sh \
  /path/to/masterdataanalysis.csv \
  /path/to/system_morphology.csv \
  /path/to/canonical_annotations.csv
```

Seven tasks are submitted simultaneously (`0-6%7`), one CPU each, with no explicit memory request.
A dependent summary job writes `MEETING_SUMMARY.md` only after all seven tasks succeed.
