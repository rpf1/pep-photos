#!/usr/bin/env python3
"""
Build a non-destructive canonical annotation export for PFPeptoids.

This does NOT edit:
- annotations.csv
- state.sqlite3
- system_images.csv
- any image
- any simulation directory

It removes only proven duplicate annotation rows from the DERIVED export:
1) historical ADA aliases -> run_0
2) D2 logical selections that reuse a canonical Cycle-1 physical observation
3) three stale Cycle-1 annotation aliases whose canonical physical observation is Cycle-2

Unresolved legacy rows are preserved but explicitly flagged.
"""

from pathlib import Path
import pandas as pd
import json

ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids")
ANALYSIS = ROOT / "analysis"
SESSION = ANALYSIS / "image_classification" / "pfp34_new_photos" / "session"
ANN = SESSION / "annotations.csv"

OUTDIR = SESSION / "canonical_export_20260929"
OUTDIR.mkdir(parents=True, exist_ok=True)

ADA_RUN = "run_First_predictions-20260813_190947_583893_ada"

# Proven physical-observation reuse / alias mappings.
# Key = stale annotation run, value = authoritative physical annotation run.
REUSE_CYCLE1 = {
    "Nk-Nfnap-Nk",
    "Nk-Nw-Nk",
    "Nk-Nw-Nwe",
    "Nm-Nse-Nfn",
    "Nn-Nfnap-Nwe",
    "Nn-Nw-Nwe",
    "Nn-Nwe-Nwe",
    "Nse-Na-Nfn",
    "Nse-Nfn-Nfn",
    "Nwe-Nw-Na",
    "Nwe-Nwe-Na",
}

CYCLE2_CANONICAL = {
    "Ne-Nm-Nfn",
    "Nk-Nk-Nwe",
    "Nse-Nm-Nfn",
}

def evidence_kind(v):
    if pd.isna(v):
        return "NONE"
    s = str(v)
    if "NEW10__" in s:
        return "STACKED"
    if "LEGACY__" in s:
        return "LEGACY"
    try:
        x = json.loads(s)
        if isinstance(x, list) and len(x) == 0:
            return "NONE"
    except Exception:
        pass
    return "OTHER"

ann = pd.read_csv(ANN)
ann["evidence_kind"] = ann["image_paths"].map(evidence_kind)
ann["canonical_include"] = True
ann["canonical_reason"] = "CURRENT_ROW"
ann["authoritative_run"] = ann["run"]

def find_counterpart(row, target_run):
    cp = ann[
        (ann["molecule"] == row["molecule"]) &
        (ann["bilayer_type"] == row["bilayer_type"]) &
        (ann["run"] == target_run) &
        (ann["evidence_kind"] == "STACKED")
    ]
    return cp

for idx, row in ann.iterrows():
    source_run = str(row["run"])
    molecule = str(row["molecule"])
    target_run = None
    reason = None

    if source_run == ADA_RUN:
        target_run = "run_0"
        reason = "HISTORICAL_ADA_ALIAS_OF_RUN0"

    elif (
        source_run == "run_ML_cycle_02"
        and molecule in REUSE_CYCLE1
        and row["evidence_kind"] in {"LEGACY", "NONE"}
    ):
        target_run = "run_ML_cycle_01"
        reason = "D2_SELECTION_REUSES_CYCLE1_PHYSICAL_OBSERVATION"

    elif (
        source_run == "run_ML_cycle_01"
        and molecule in CYCLE2_CANONICAL
        and row["evidence_kind"] in {"LEGACY", "NONE"}
    ):
        target_run = "run_ML_cycle_02"
        reason = "STALE_CYCLE1_ALIAS_CANONICAL_PHYSICAL_OBSERVATION_IS_CYCLE2"

    if target_run is not None:
        cp = find_counterpart(row, target_run)
        if len(cp) != 1:
            raise RuntimeError(
                f"Expected exactly one STACKED counterpart for "
                f"{molecule} | {row['bilayer_type']} | {source_run} -> {target_run}; "
                f"found {len(cp)}"
            )
        ann.at[idx, "canonical_include"] = False
        ann.at[idx, "canonical_reason"] = reason
        ann.at[idx, "authoritative_run"] = target_run

# Explicit unresolved rows retained for provenance but not primary STACKED analysis.
mask_nkeqm = (
    (ann["molecule"] == "Nf-Nf-Nkeqm") &
    (ann["run"] == "run_ML_cycle_01") &
    (ann["evidence_kind"] != "STACKED")
)
ann.loc[mask_nkeqm, "canonical_reason"] = "NEEDS_EXACT_STACKED_REGENERATION_AND_REVIEW"

mask_nknns = (
    (ann["molecule"] == "Nk-Nn-Ns") &
    (ann["run"] == "run_0") &
    (ann["evidence_kind"] != "STACKED")
)
ann.loc[mask_nknns, "canonical_reason"] = "LEGACY_ONLY_NO_CURRENT_SIMULATION_SOURCE"

canonical = ann[ann["canonical_include"]].copy()

primary = canonical[
    (canonical["annotation_status"] == "labelled") &
    (canonical["evidence_kind"] == "STACKED")
].copy()

unresolved = canonical[
    canonical["canonical_reason"].isin({
        "NEEDS_EXACT_STACKED_REGENERATION_AND_REVIEW",
        "LEGACY_ONLY_NO_CURRENT_SIMULATION_SOURCE",
    })
].copy()

ann.to_csv(OUTDIR / "all_annotations_with_canonical_decision.csv", index=False)
canonical.to_csv(OUTDIR / "canonical_annotations.csv", index=False)
primary.to_csv(OUTDIR / "primary_stacked_labelled_annotations.csv", index=False)
unresolved.to_csv(OUTDIR / "unresolved_legacy_annotations.csv", index=False)

summary = {
    "original_rows": int(len(ann)),
    "proven_duplicate_rows_excluded_from_canonical_export": int((~ann["canonical_include"]).sum()),
    "canonical_rows": int(len(canonical)),
    "primary_stacked_labelled_rows": int(len(primary)),
    "unresolved_legacy_rows": int(len(unresolved)),
    "unresolved_by_reason": unresolved["canonical_reason"].value_counts().to_dict(),
}
(OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
print(f"Saved: {OUTDIR}")
