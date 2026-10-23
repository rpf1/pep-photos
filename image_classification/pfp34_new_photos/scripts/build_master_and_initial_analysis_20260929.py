#!/usr/bin/env python3
"""
PFPeptoids canonical manual + simulation master builder and initial analysis.

READS ONLY from project/pilot sources.
WRITES ONLY to a new output directory and a ZIP beside it.
Never edits annotations.csv, state.sqlite3, masterdataanalysis.csv, simulation
folders, photographs, ML state, or existing morphology results.

Main deliverables:
  01_system_master_extended.csv
  02_system_master_core.csv
  03_paired_three_row_all.csv
  04_tripeptoid_three_row_all.csv
  05_tripeptoid_three_row_primary.csv
  06_tripeptoid_pairs_wide_primary.csv
  07_popc80pops20_systems.csv
  08_metric_dictionary.csv
  09_join_coverage.csv
  10_annotation_concentration_ambiguities.csv
  11_unmatched_annotations.csv
  12_unmatched_simulations.csv
  13_run_metadata_path_mismatches.csv
  14_label_counts.csv
  15_pair_label_counts.csv
  16_ddapl_by_label_pair.csv
  17_metric_summary_by_label.csv
  18_numeric_correlations.csv
  19_target_candidate_summary.csv
  20_pilot_source_coverage.csv
  21_photo_exceptions.csv
  support_*.csv
  VALIDATION.json
  README.md
  master_analysis_20260929_v1.zip
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import zipfile
import warnings

# Keep terminal output focused on scientific/data validation. These are pandas
# API deprecation notices only; real exceptions and runtime warnings are NOT hidden.
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

MAMMALIAN = "POPC60CHOL40_W_WF"
BACTERIAL = "POPE75POPG25_W_WF"
POPS = "POPC80POPS20_W_WF"

MEMBRANE_TO_BILAYER = {
    "mammalian": MAMMALIAN,
    "bacterial": BACTERIAL,
}

IDENTITY_ANN_COLS = {
    "system_key": "annotation_system_key",
    "label": "manual_label",
    "annotation_status": "annotation_status",
    "note": "annotation_note",
    "annotation_timestamp": "annotation_timestamp",
    "last_modified": "annotation_last_modified",
    "image_count": "annotation_image_count",
    "image_paths": "annotation_image_paths",
    "evidence_sha256": "annotation_evidence_sha256",
    "config_sha256": "annotation_config_sha256",
    "event_id": "annotation_event_id",
    "evidence_kind": "annotation_evidence_kind",
    "canonical_reason": "annotation_canonical_reason",
    "authoritative_run": "annotation_authoritative_run",
}

SIM_IDENTITY_COLS = [
    "molecule_type", "component_count", "component_names", "component_counts",
    "component_types", "component_token_sequences", "component_lengths",
    "component_molecular_weights", "component_missing_mass_codes",
    "component_positive_charges", "component_negative_charges",
    "component_net_charges", "contains_mixture", "length", "molecular_weight",
    "molecular_weight_missing_codes", "positive_charge", "negative_charge",
    "net_charge", "charge_density",
]

SIM_RESPONSE_COLS = [
    "start_APL_BOX", "finish_APL_BOX_100ns", "APL_change_BOX",
    "APL_percent_change_BOX", "APL_mean_last10ns_BOX", "APL_std_last10ns_BOX",
    "start_thickness", "finish_thickness_100ns", "thickness_change",
    "membrane_area_start", "membrane_area_finish",
    "lipids_per_leaflet", "n_phospholipids", "n_chol",
    "n_membrane_molecules",
    "final_relax_time_ns", "timepoint_100ns_found", "analysis_complete",
    "relax_end_time_ns", "reached_100ns", "target_relax_time_ns",
    "remaining_relax_time_ns",
]

# PO4-derived APL is retained only as explicitly noncanonical QC/reference.
PO4_QC_COLS = [
    "start_APL_PO4", "finish_APL_PO4_100ns", "APL_change_PO4",
    "APL_percent_change_PO4", "APL_mean_last10ns_PO4",
    "APL_std_last10ns_PO4",
]

MORPH_CORE_SOURCE_COLS = [
    "baseline__midplane_excess_area_fraction",
    "endpoint__midplane_excess_area_fraction",
    "delta__midplane_excess_area_fraction",
    "baseline__height_rms_nm",
    "endpoint__height_rms_nm",
    "delta__height_rms_nm",
    "baseline__H_rms_nm_inv",
    "endpoint__H_rms_nm_inv",
    "delta__H_rms_nm_inv",
    "baseline__thickness_median_nm",
    "endpoint__thickness_median_nm",
    "delta__thickness_median_nm",
    "baseline__thickness_valid_fraction",
    "endpoint__thickness_valid_fraction",
    "baseline__fit_rmse_upper_nm",
    "endpoint__fit_rmse_upper_nm",
    "baseline__fit_rmse_lower_nm",
    "endpoint__fit_rmse_lower_nm",
    "reproduced_projected_dAPL_A2",
    "projected_dAPL_error_vs_stored_A2",
    "relax_90_100__midplane_excess_area_fraction__median",
    "relax_90_100__height_rms_nm__median",
    "relax_90_100__H_rms_nm_inv__median",
    "relax_90_100__thickness_median_nm__median",
    "relax_30_100__midplane_excess_area_fraction__slope_per_ns",
    "relax_30_100__height_rms_nm__slope_per_ns",
    "relax_30_100__H_rms_nm_inv__slope_per_ns",
    "relax_30_100__thickness_median_nm__slope_per_ns",
]

LEAFLET_SMOKE_SOURCE_COLS = [
    "endpoint__upper_surface_area_nm2",
    "endpoint__lower_surface_area_nm2",
    "endpoint__upper_excess_area_fraction",
    "endpoint__lower_excess_area_fraction",
    "endpoint__upper_height_rms_nm",
    "endpoint__lower_height_rms_nm",
    "endpoint__upper_H_rms_nm_inv",
    "endpoint__lower_H_rms_nm_inv",
    "endpoint__surface_area_upper_minus_lower_nm2",
    "endpoint__surface_area_abs_upper_minus_lower_nm2",
    "endpoint__excess_area_upper_minus_lower",
    "endpoint__excess_area_abs_upper_minus_lower",
    "endpoint__height_rms_upper_minus_lower_nm",
    "endpoint__height_rms_abs_upper_minus_lower_nm",
    "endpoint__H_rms_upper_minus_lower_nm_inv",
    "endpoint__H_rms_abs_upper_minus_lower_nm_inv",
    "endpoint__PO4_upper_count",
    "endpoint__PO4_lower_count",
    "analysed_frames",
    "review_frames",
    "PO4_assignment_changes_total",
    "CHOL_assignment_changes_total",
    "CHOL_intervals_with_changes",
    "CHOL_upper_count_min",
    "CHOL_upper_count_max",
    "CHOL_lower_count_min",
    "CHOL_lower_count_max",
    "CHOL_near_midplane_count_max",
    "CHOL_frames_near_midplane",
    "PO4_local_midplane_disagreement_max",
    "baseline_qc_flags",
]

# Same metric means across membranes, so these are permitted in the third
# bacterial-minus-mammalian row. Species-specific lipid metrics are not.
PAIR_DIFFERENCE_COLS = [
    "APL_change_BOX",
    "APL_percent_change_BOX",
    "thickness_change",
    "morph__endpoint__midplane_excess_area_fraction",
    "morph__delta__midplane_excess_area_fraction",
    "morph__endpoint__height_rms_nm",
    "morph__delta__height_rms_nm",
    "morph__endpoint__H_rms_nm_inv",
    "morph__delta__H_rms_nm_inv",
    "morph__endpoint__thickness_median_nm",
    "morph__delta__thickness_median_nm",
    "morph__relax_90_100__midplane_excess_area_fraction__median",
    "morph__relax_90_100__height_rms_nm__median",
    "morph__relax_90_100__H_rms_nm_inv__median",
    "morph__relax_90_100__thickness_median_nm__median",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids",
    )
    p.add_argument(
        "--morph-root",
        default="/users/pfb19164/pfp_morphology_pilot",
    )
    p.add_argument("--canonical-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--bilayer-output-dir", default=None)
    return p.parse_args()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_nmols(value) -> list[int]:
    if pd.isna(value):
        return []
    return sorted(set(int(x) for x in re.findall(r"__nmol_(\d+)__", str(value))))

def infer_type_and_length(molecule: str):
    s = str(molecule)
    parts = s.split("-")
    if len(parts) >= 1 and all(p.startswith("N") for p in parts):
        return "peptoid", len(parts)
    if "-" not in s and re.fullmatch(r"[A-Z]+", s):
        return "peptide", len(s)
    return np.nan, np.nan

def physical_run_from_system_dir(value):
    if pd.isna(value):
        return np.nan
    m = re.search(r"/(run_[^/]+)/[^/]+/?$", str(value))
    return m.group(1) if m else np.nan

def boolish(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )

def membrane_role(bilayer):
    return {
        MAMMALIAN: "mammalian_like",
        BACTERIAL: "bacterial_like",
        POPS: "POPC80POPS20",
    }.get(str(bilayer), "other")

def add_metric_dictionary(rows, metric, family, units, source, status,
                          paired_difference_allowed, notes=""):
    rows.append({
        "metric": metric,
        "family": family,
        "units": units,
        "source": source,
        "status": status,
        "paired_difference_allowed": paired_difference_allowed,
        "notes": notes,
    })

def metric_units(name):
    n = name.lower()
    if "nm_inv" in n:
        return "nm^-1"
    if n.endswith("_nm") or "_nm__" in n or "_nm_" in n:
        return "nm"
    if "area_nm2" in n or "nm2" in n:
        return "nm^2"
    if "apl" in n and ("a2" in n or "box" in n or "po4" in n):
        return "A^2 per membrane molecule"
    if "density" in n and "molecules_nm2" in n:
        return "molecules/nm^2"
    if "fraction" in n or "enrichment" in n or "corr_" in n:
        return "dimensionless"
    if "percent" in n:
        return "%"
    if "count" in n or "frames" in n or "intervals" in n:
        return "count"
    return ""

def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

def safe_numeric_summary(df, group_cols, metrics):
    records = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce")
        temp = df[group_cols].copy()
        temp["_v"] = vals
        temp = temp[temp["_v"].notna()]
        if temp.empty:
            continue
        for key, g in temp.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            rec = dict(zip(group_cols, key))
            v = g["_v"]
            rec.update({
                "metric": metric,
                "n": int(v.count()),
                "mean": v.mean(),
                "std": v.std(ddof=1),
                "median": v.median(),
                "q25": q25(v),
                "q75": q75(v),
                "min": v.min(),
                "max": v.max(),
            })
            records.append(rec)
    return pd.DataFrame(records)

def spearman_long(df, metrics, dataset_name, min_n=3):
    records = []
    available = [m for m in metrics if m in df.columns]
    numeric = df[available].apply(pd.to_numeric, errors="coerce")
    for i, a in enumerate(available):
        for b in available[i + 1:]:
            mask = numeric[a].notna() & numeric[b].notna()
            n = int(mask.sum())
            if n < min_n:
                continue
            rho = numeric.loc[mask, a].corr(numeric.loc[mask, b], method="spearman")
            records.append({
                "dataset": dataset_name,
                "metric_a": a,
                "metric_b": b,
                "n": n,
                "spearman_rho": rho,
            })
    return pd.DataFrame(records)

def main():
    args = parse_args()
    root = Path(args.root)
    analysis = root / "analysis"
    morph_root = Path(args.morph_root)
    session = analysis / "image_classification" / "pfp34_new_photos" / "session"
    canonical_dir = (
        Path(args.canonical_dir)
        if args.canonical_dir
        else session / "canonical_export_20260929"
    )
    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else session / "master_analysis_20260929_v1"
    )
    zip_path = outdir.with_suffix(".zip")
    bilayer_output_dir = (
        Path(args.bilayer_output_dir)
        if args.bilayer_output_dir
        else analysis / "bilayer_output"
    )

    if outdir.exists() or zip_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing output:\n{outdir}\n{zip_path}"
        )

    ann_path = canonical_dir / "canonical_annotations.csv"
    primary_ann_path = canonical_dir / "primary_stacked_labelled_annotations.csv"
    unresolved_path = canonical_dir / "unresolved_legacy_annotations.csv"
    canon_summary_path = canonical_dir / "summary.json"
    master_path = analysis / "masterdataanalysis.csv"

    required = [ann_path, primary_ann_path, unresolved_path, canon_summary_path, master_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    outdir.mkdir(parents=True)

    source_hashes = {str(p): sha256(p) for p in required}

    # ------------------------------------------------------------------
    # 1. Canonical annotations
    # ------------------------------------------------------------------
    ann = pd.read_csv(ann_path, low_memory=False)
    ann["_nmol_values"] = ann["image_paths"].map(parse_nmols)
    ann["annotation_nmol_values"] = ann["_nmol_values"].map(
        lambda x: ";".join(str(v) for v in x)
    )
    ann["annotation_concentration_ambiguous"] = ann["_nmol_values"].map(len) != 1
    ann["nmol"] = ann["_nmol_values"].map(
        lambda x: x[0] if len(x) == 1 else np.nan
    )
    ann["physical_run"] = ann["run"].astype(str)
    ann["molecule"] = ann["molecule"].astype(str)
    ann["bilayer_type"] = ann["bilayer_type"].astype(str)

    inferred = ann["molecule"].map(infer_type_and_length)
    ann["annotation_inferred_molecule_type"] = inferred.map(lambda x: x[0])
    ann["annotation_inferred_length"] = inferred.map(lambda x: x[1])

    ambiguities = ann[ann["annotation_concentration_ambiguous"]].copy()
    ambiguities.drop(columns=["_nmol_values"], inplace=True)
    ambiguities.to_csv(outdir / "10_annotation_concentration_ambiguities.csv", index=False)

    ann_matchable = ann[~ann["annotation_concentration_ambiguous"]].copy()
    ann_matchable["nmol"] = ann_matchable["nmol"].astype(int)
    ann_matchable["physical_key"] = (
        ann_matchable["molecule"] + "|" +
        ann_matchable["nmol"].astype(str) + "|" +
        ann_matchable["bilayer_type"] + "|" +
        ann_matchable["physical_run"]
    )
    if ann_matchable["physical_key"].duplicated().any():
        dup = ann_matchable.loc[
            ann_matchable["physical_key"].duplicated(keep=False),
            ["physical_key", "system_key", "molecule", "bilayer_type", "run"]
        ]
        dup.to_csv(outdir / "FATAL_duplicate_annotation_keys.csv", index=False)
        raise RuntimeError("Duplicate matchable canonical annotation physical keys.")

    # ------------------------------------------------------------------
    # 2. Canonical simulation/APL table
    # ------------------------------------------------------------------
    sim = pd.read_csv(master_path, low_memory=False)
    sim["molecule"] = sim["token_sequence"].astype(str)
    sim["bilayer_type"] = sim["system_type"].astype(str)
    sim["nmol"] = pd.to_numeric(sim["nmol"], errors="raise").astype(int)

    sim["physical_run"] = sim["system_dir"].map(physical_run_from_system_dir)
    fallback_run = "run_" + sim["run_number"].astype(str)
    sim["physical_run"] = sim["physical_run"].fillna(fallback_run)
    sim["master_run_metadata"] = fallback_run
    sim["run_metadata_path_mismatch"] = (
        sim["physical_run"] != sim["master_run_metadata"]
    )

    sim["physical_key"] = (
        sim["molecule"] + "|" +
        sim["nmol"].astype(str) + "|" +
        sim["bilayer_type"] + "|" +
        sim["physical_run"]
    )
    if sim["physical_key"].duplicated().any():
        dup = sim.loc[
            sim["physical_key"].duplicated(keep=False),
            ["physical_key", "system_key", "system_dir", "run_number"]
        ]
        dup.to_csv(outdir / "FATAL_duplicate_simulation_keys.csv", index=False)
        raise RuntimeError("Duplicate physical simulation keys in masterdataanalysis.csv.")

    # Canonical BOX identity check.
    sim["_apl_recalc"] = (
        pd.to_numeric(sim["finish_APL_BOX_100ns"], errors="coerce")
        - pd.to_numeric(sim["start_APL_BOX"], errors="coerce")
    )
    sim["_apl_error"] = (
        pd.to_numeric(sim["APL_change_BOX"], errors="coerce") - sim["_apl_recalc"]
    )

    run_mismatches = sim[sim["run_metadata_path_mismatch"]].copy()
    run_mismatches.to_csv(outdir / "13_run_metadata_path_mismatches.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Exact annotation ↔ simulation merge, plus unmatched records
    # ------------------------------------------------------------------
    ann_for_merge = ann_matchable.copy()
    rename = {
        k: v for k, v in IDENTITY_ANN_COLS.items()
        if k in ann_for_merge.columns
    }
    ann_for_merge = ann_for_merge.rename(columns=rename)

    keep_ann = [
        "physical_key", "molecule", "nmol", "bilayer_type", "physical_run",
        "annotation_nmol_values", "annotation_concentration_ambiguous",
        "annotation_inferred_molecule_type", "annotation_inferred_length",
    ] + list(rename.values())
    keep_ann = [c for c in keep_ann if c in ann_for_merge.columns]
    ann_for_merge = ann_for_merge[keep_ann]

    merged = sim.merge(
        ann_for_merge.drop(
            columns=["molecule", "nmol", "bilayer_type", "physical_run"],
            errors="ignore",
        ),
        on="physical_key",
        how="outer",
        indicator="_join",
    )

    # Recover identity on annotation-only rows from physical_key.
    key_parts = merged["physical_key"].astype(str).str.split("|", expand=True)
    merged["molecule"] = merged["molecule"].fillna(key_parts[0])
    merged["nmol"] = pd.to_numeric(
        merged["nmol"].fillna(pd.to_numeric(key_parts[1], errors="coerce")),
        errors="coerce",
    )
    merged["bilayer_type"] = merged["bilayer_type"].fillna(key_parts[2])
    merged["physical_run"] = merged["physical_run"].fillna(key_parts[3])

    # Annotation-only rows need inferred type/length.
    merged["molecule_type"] = merged["molecule_type"].fillna(
        merged.get("annotation_inferred_molecule_type")
    )
    merged["length"] = pd.to_numeric(merged["length"], errors="coerce").fillna(
        pd.to_numeric(merged.get("annotation_inferred_length"), errors="coerce")
    )

    merged["membrane_role"] = merged["bilayer_type"].map(membrane_role)
    merged["join_status"] = merged["_join"].map({
        "both": "annotation_and_simulation",
        "left_only": "simulation_only",
        "right_only": "annotation_only",
    }).astype(str)

    # Add the concentration-ambiguous annotation records as provenance-only rows.
    if not ambiguities.empty:
        amb = pd.DataFrame(index=range(len(ambiguities)), columns=merged.columns)
        amb["physical_key"] = (
            ambiguities["molecule"].astype(str) + "|AMBIGUOUS_NMOL|" +
            ambiguities["bilayer_type"].astype(str) + "|" +
            ambiguities["run"].astype(str)
        ).values
        amb["molecule"] = ambiguities["molecule"].values
        amb["bilayer_type"] = ambiguities["bilayer_type"].values
        amb["physical_run"] = ambiguities["run"].values
        amb["membrane_role"] = ambiguities["bilayer_type"].map(membrane_role).values
        amb["molecule_type"] = ambiguities["annotation_inferred_molecule_type"].values
        amb["length"] = ambiguities["annotation_inferred_length"].values
        amb["annotation_nmol_values"] = ambiguities["annotation_nmol_values"].values
        amb["annotation_concentration_ambiguous"] = True
        for src, dst in rename.items():
            if src in ambiguities and dst in amb:
                amb[dst] = ambiguities[src].values
        amb["join_status"] = "annotation_concentration_ambiguous"
        merged = pd.concat([merged, amb], ignore_index=True)

    merged["annotation_concentration_ambiguous"] = (
        merged["annotation_concentration_ambiguous"].fillna(False).astype(bool)
    )

    merged["annotation_primary_label_eligible"] = (
        (merged.get("annotation_status") == "labelled")
        & (merged.get("annotation_evidence_kind") == "STACKED")
        & (~merged["annotation_concentration_ambiguous"])
    )

    merged["simulation_complete_100ns"] = (
        boolish(merged["analysis_complete"])
        & boolish(merged["reached_100ns"])
        & boolish(merged["timepoint_100ns_found"])
    )

    merged["primary_system_eligible"] = (
        merged["annotation_primary_label_eligible"]
        & merged["simulation_complete_100ns"]
        & (merged["join_status"] == "annotation_and_simulation")
    )

    # ------------------------------------------------------------------
    # 4. Morphology/local-lipid pilot joins
    # ------------------------------------------------------------------
    pilot_files = {
        "balanced_panel_systems": morph_root / "results" / "balanced_panel_systems.csv",
        "morphology": morph_root / "results" / "morphology_v1" / "grid32_stride10_lambda4nm" / "morphology_v1_summary.csv",
        "lipid_system": morph_root / "results" / "lipid_state_v1" / "grid32_stride10_lambda4nm_sigma1nm" / "lipid_state_v1_system_summary.csv",
        "lipid_leaflet": morph_root / "results" / "lipid_state_v1" / "grid32_stride10_lambda4nm_sigma1nm" / "lipid_state_v1_leaflet_summary.csv",
        "leaflet_smoke_summary": morph_root / "results" / "morphology_leaflets" / "smoke" / "summary.csv",
        "leaflet_smoke_qc": morph_root / "results" / "morphology_leaflets" / "smoke" / "qc.csv",
        "leaflet_dense_summary": morph_root / "results" / "morphology_leaflets" / "chol_dense_1ns" / "summary.csv",
        "leaflet_dense_qc": morph_root / "results" / "morphology_leaflets" / "chol_dense_1ns" / "qc.csv",
        "metric_qc": morph_root / "results" / "metric_selection_v1" / "qc_summary.csv",
        "smoothing": morph_root / "results" / "metric_selection_v1" / "smoothing_stability_summary.csv",
    }

    pilot_coverage = []

    if pilot_files["balanced_panel_systems"].exists():
        bps = pd.read_csv(pilot_files["balanced_panel_systems"])
        bps["bilayer_type"] = bps["membrane"].map(MEMBRANE_TO_BILAYER)
        bps["nmol"] = pd.to_numeric(
            bps["system_dir"].astype(str).str.extract(r"_(\d+)$")[0],
            errors="raise",
        ).astype(int)
        bps["physical_key"] = (
            bps["id"].astype(str) + "|" +
            bps["nmol"].astype(str) + "|" +
            bps["bilayer_type"].astype(str) + "|" +
            bps["source_run"].astype(str)
        )
        if bps["physical_key"].duplicated().any():
            raise RuntimeError("Duplicate physical keys in balanced_panel_systems.csv")
        bps_map = bps[["id", "membrane", "physical_key", "source_run", "panel_role"]].copy()
        source_hashes[str(pilot_files["balanced_panel_systems"])] = sha256(
            pilot_files["balanced_panel_systems"]
        )
    else:
        bps = pd.DataFrame()
        bps_map = pd.DataFrame(columns=["id", "membrane", "physical_key", "source_run", "panel_role"])

    def map_pilot_identity(df):
        out = df.merge(
            bps_map[["id", "membrane", "physical_key"]],
            on=["id", "membrane"],
            how="left",
            validate="many_to_one",
        )
        return out

    # Main validated/core 4 nm morphology.
    morph_selected = pd.DataFrame()
    if pilot_files["morphology"].exists():
        morph = pd.read_csv(pilot_files["morphology"], low_memory=False)
        morph = map_pilot_identity(morph)
        missing_keys = int(morph["physical_key"].isna().sum())
        selected = ["physical_key"] + [
            c for c in MORPH_CORE_SOURCE_COLS if c in morph.columns
        ]
        morph_selected = morph[selected].copy()
        morph_selected = morph_selected.rename(
            columns={c: "morph__" + c for c in selected if c != "physical_key"}
        )
        morph_selected["morphology_available"] = True
        morph_selected["morphology_method"] = "grid32_stride10_lambda4nm"
        morph_selected["morphology_status"] = "validated read-only pilot; not production APL"
        merged = merged.merge(morph_selected, on="physical_key", how="left", validate="one_to_one")
        pilot_coverage.append({
            "source": "morphology_4nm_core",
            "source_rows": len(morph),
            "unmapped_source_rows": missing_keys,
            "matched_system_master_rows": int(merged["morphology_available"].fillna(False).sum()),
        })
        source_hashes[str(pilot_files["morphology"])] = sha256(pilot_files["morphology"])

    # Local lipid system summary: retain all numeric/scientific columns, prefixed.
    lipid_system_selected = pd.DataFrame()
    if pilot_files["lipid_system"].exists():
        lip = pd.read_csv(pilot_files["lipid_system"], low_memory=False)
        lip = map_pilot_identity(lip)
        identity = {"id", "panel_role", "membrane", "ddapl", "stored_dapl", "physical_key"}
        metric_cols = [c for c in lip.columns if c not in identity]
        lipid_system_selected = lip[["physical_key"] + metric_cols].copy()
        lipid_system_selected = lipid_system_selected.rename(
            columns={c: "lipid_system__" + c for c in metric_cols}
        )
        lipid_system_selected["local_lipid_system_available"] = True
        lipid_system_selected["local_lipid_method_status"] = (
            "read-only pilot; periodic XY Gaussian smoothing; fitted-surface Jacobian; not geodesic density"
        )
        merged = merged.merge(
            lipid_system_selected, on="physical_key", how="left", validate="one_to_one"
        )
        pilot_coverage.append({
            "source": "local_lipid_system",
            "source_rows": len(lip),
            "unmapped_source_rows": int(lip["physical_key"].isna().sum()),
            "matched_system_master_rows": int(
                merged["local_lipid_system_available"].fillna(False).sum()
            ),
        })
        source_hashes[str(pilot_files["lipid_system"])] = sha256(pilot_files["lipid_system"])

    # Local lipid leaflet summary: pivot upper/lower into one system row.
    lipid_leaflet_pivot = pd.DataFrame()
    if pilot_files["lipid_leaflet"].exists():
        leaf = pd.read_csv(pilot_files["lipid_leaflet"], low_memory=False)
        leaf = map_pilot_identity(leaf)
        identity = {"id", "panel_role", "membrane", "leaflet", "ddapl", "stored_dapl", "physical_key"}
        metric_cols = [c for c in leaf.columns if c not in identity]
        pieces = []
        for leaflet in ["upper", "lower"]:
            part = leaf[leaf["leaflet"] == leaflet][["physical_key"] + metric_cols].copy()
            part = part.rename(
                columns={c: f"lipid_{leaflet}__{c}" for c in metric_cols}
            )
            pieces.append(part)
        if pieces:
            lipid_leaflet_pivot = pieces[0]
            for part in pieces[1:]:
                lipid_leaflet_pivot = lipid_leaflet_pivot.merge(
                    part, on="physical_key", how="outer", validate="one_to_one"
                )
            lipid_leaflet_pivot["local_lipid_leaflets_available"] = True
            merged = merged.merge(
                lipid_leaflet_pivot, on="physical_key", how="left", validate="one_to_one"
            )
        pilot_coverage.append({
            "source": "local_lipid_leaflets",
            "source_rows": len(leaf),
            "unmapped_source_rows": int(leaf["physical_key"].isna().sum()),
            "matched_system_master_rows": int(
                merged.get(
                    "local_lipid_leaflets_available",
                    pd.Series(False, index=merged.index),
                ).fillna(False).sum()
            ),
        })
        source_hashes[str(pilot_files["lipid_leaflet"])] = sha256(pilot_files["lipid_leaflet"])

    # Explicit upper/lower smoke geometry: selected core/asymmetry metrics.
    smoke_selected = pd.DataFrame()
    if pilot_files["leaflet_smoke_summary"].exists():
        smoke = pd.read_csv(pilot_files["leaflet_smoke_summary"], low_memory=False)
        smoke = map_pilot_identity(smoke)
        selected = ["physical_key"] + [
            c for c in LEAFLET_SMOKE_SOURCE_COLS if c in smoke.columns
        ]
        smoke_selected = smoke[selected].copy()
        smoke_selected = smoke_selected.rename(
            columns={c: "leaflet_smoke__" + c for c in selected if c != "physical_key"}
        )
        smoke_selected["leaflet_smoke_available"] = True

        # Add frame-level QC counts without treating frames as replicates.
        if pilot_files["leaflet_smoke_qc"].exists():
            sqc = pd.read_csv(pilot_files["leaflet_smoke_qc"], low_memory=False)
            sqc = map_pilot_identity(sqc)
            sqc["_flagged"] = sqc["qc_flags"].fillna("").astype(str).str.len() > 0
            qagg = sqc.groupby("physical_key", dropna=False).agg(
                leaflet_smoke__qc_frames=("frame", "size"),
                leaflet_smoke__qc_flagged_frames=("_flagged", "sum"),
            ).reset_index()
            qagg["leaflet_smoke__qc_flagged_fraction"] = (
                qagg["leaflet_smoke__qc_flagged_frames"]
                / qagg["leaflet_smoke__qc_frames"]
            )
            smoke_selected = smoke_selected.merge(
                qagg, on="physical_key", how="left", validate="one_to_one"
            )
            source_hashes[str(pilot_files["leaflet_smoke_qc"])] = sha256(
                pilot_files["leaflet_smoke_qc"]
            )

        merged = merged.merge(smoke_selected, on="physical_key", how="left", validate="one_to_one")
        pilot_coverage.append({
            "source": "explicit_leaflet_smoke",
            "source_rows": len(smoke),
            "unmapped_source_rows": int(smoke["physical_key"].isna().sum()),
            "matched_system_master_rows": int(
                merged["leaflet_smoke_available"].fillna(False).sum()
            ),
        })
        source_hashes[str(pilot_files["leaflet_smoke_summary"])] = sha256(
            pilot_files["leaflet_smoke_summary"]
        )

    # Dense CHOL run: QC only, explicitly separate from smoke geometry.
    dense_qc_selected = pd.DataFrame()
    if pilot_files["leaflet_dense_summary"].exists():
        dense = pd.read_csv(pilot_files["leaflet_dense_summary"], low_memory=False)
        dense = map_pilot_identity(dense)
        dense_cols = [
            "analysed_frames", "review_frames",
            "CHOL_observed_intervals", "CHOL_assignment_changes_total",
            "CHOL_intervals_with_changes", "CHOL_upper_count_min",
            "CHOL_upper_count_max", "CHOL_lower_count_min",
            "CHOL_lower_count_max", "CHOL_near_midplane_count_max",
            "CHOL_frames_near_midplane", "baseline_qc_flags",
        ]
        dense_cols = [c for c in dense_cols if c in dense.columns]
        dense_qc_selected = dense[["physical_key"] + dense_cols].copy()
        dense_qc_selected = dense_qc_selected.rename(
            columns={c: "dense_chol_qc__" + c for c in dense_cols}
        )
        dense_qc_selected["dense_chol_qc_available"] = True
        dense_qc_selected["dense_chol_qc_sampling"] = "1 ns; overlapping pilot, not independent replicate"

        if pilot_files["leaflet_dense_qc"].exists():
            dqc = pd.read_csv(pilot_files["leaflet_dense_qc"], low_memory=False)
            dqc = map_pilot_identity(dqc)
            dqc["_flagged"] = dqc["qc_flags"].fillna("").astype(str).str.len() > 0
            qagg = dqc.groupby("physical_key", dropna=False).agg(
                dense_chol_qc__frames=("frame", "size"),
                dense_chol_qc__flagged_frames=("_flagged", "sum"),
            ).reset_index()
            qagg["dense_chol_qc__flagged_fraction"] = (
                qagg["dense_chol_qc__flagged_frames"]
                / qagg["dense_chol_qc__frames"]
            )
            dense_qc_selected = dense_qc_selected.merge(
                qagg, on="physical_key", how="left", validate="one_to_one"
            )
            source_hashes[str(pilot_files["leaflet_dense_qc"])] = sha256(
                pilot_files["leaflet_dense_qc"]
            )

        merged = merged.merge(
            dense_qc_selected, on="physical_key", how="left", validate="one_to_one"
        )
        pilot_coverage.append({
            "source": "dense_CHOL_QC",
            "source_rows": len(dense),
            "unmapped_source_rows": int(dense["physical_key"].isna().sum()),
            "matched_system_master_rows": int(
                merged["dense_chol_qc_available"].fillna(False).sum()
            ),
        })
        source_hashes[str(pilot_files["leaflet_dense_summary"])] = sha256(
            pilot_files["leaflet_dense_summary"]
        )

    # Explicit caution flags.
    merged["local_lipid_pilot_caution"] = np.where(
        merged.get(
            "local_lipid_system_available",
            pd.Series(False, index=merged.index),
        ).fillna(False),
        "PILOT_NOT_PRODUCTION; local-density interpretation requires method/QC context",
        "",
    )
    merged["chol_leaflet_interpretation_caution"] = np.where(
        (merged["bilayer_type"] == MAMMALIAN)
        & (
            merged.get(
                "local_lipid_system_available",
                pd.Series(False, index=merged.index),
            ).fillna(False)
            | merged.get(
                "leaflet_smoke_available",
                pd.Series(False, index=merged.index),
            ).fillna(False)
        ),
        "CHOL leaflet assignment has near-midplane/switching QC; do not infer redistribution unqualified",
        "",
    )

    # ------------------------------------------------------------------
    # 5. System master outputs
    # ------------------------------------------------------------------
    merged = merged.drop(columns=["_join", "_apl_recalc", "_apl_error"], errors="ignore")

    # Stable leading order.
    leading = [
        "physical_key", "molecule", "molecule_type", "length", "nmol",
        "bilayer_type", "membrane_role", "physical_run", "join_status",
        "annotation_system_key", "manual_label", "annotation_status",
        "annotation_evidence_kind", "annotation_note",
        "annotation_primary_label_eligible", "simulation_complete_100ns",
        "primary_system_eligible",
    ]
    leading = [c for c in leading if c in merged.columns]
    rest = [c for c in merged.columns if c not in leading]
    merged = merged[leading + rest].sort_values(
        ["molecule", "nmol", "physical_run", "bilayer_type"],
        na_position="last",
    )

    merged.to_csv(outdir / "01_system_master_extended.csv", index=False)

    core_candidates = (
        leading
        + SIM_IDENTITY_COLS
        + SIM_RESPONSE_COLS
        + [f"qc_PO4__{c}" for c in PO4_QC_COLS]
        + [c for c in merged.columns if c.startswith("morph__")]
        + [
            c for c in merged.columns
            if c.startswith("leaflet_smoke__")
            or c.startswith("dense_chol_qc__")
        ]
        + [
            "morphology_available", "morphology_method",
            "local_lipid_system_available", "local_lipid_leaflets_available",
            "leaflet_smoke_available", "dense_chol_qc_available",
            "local_lipid_pilot_caution", "chol_leaflet_interpretation_caution",
            "system_dir", "output_dir", "master_run_metadata",
            "run_metadata_path_mismatch",
        ]
    )

    # Rename PO4 QC fields only in the core view, never source data.
    core = merged.copy()
    po4_rename = {c: f"qc_PO4__{c}" for c in PO4_QC_COLS if c in core.columns}
    core = core.rename(columns=po4_rename)
    core_cols = []
    for c in core_candidates:
        if c in core.columns and c not in core_cols:
            core_cols.append(c)
    core[core_cols].to_csv(outdir / "02_system_master_core.csv", index=False)

    unmatched_ann = merged[
        merged["join_status"].isin(
            ["annotation_only", "annotation_concentration_ambiguous"]
        )
    ].copy()
    unmatched_ann.to_csv(outdir / "11_unmatched_annotations.csv", index=False)

    unmatched_sim = merged[merged["join_status"] == "simulation_only"].copy()
    unmatched_sim.to_csv(outdir / "12_unmatched_simulations.csv", index=False)

    # POPS remains available separately, never silently pooled into the pair.
    merged[merged["bilayer_type"] == POPS].to_csv(
        outdir / "07_popc80pops20_systems.csv", index=False
    )

    # ------------------------------------------------------------------
    # 6. Three-row paired outputs
    # ------------------------------------------------------------------
    pair_base = merged[
        (merged["join_status"] != "annotation_concentration_ambiguous")
        & merged["bilayer_type"].isin([MAMMALIAN, BACTERIAL])
        & merged["nmol"].notna()
    ].copy()

    pair_base["pair_key"] = (
        pair_base["molecule"].astype(str) + "|" +
        pair_base["nmol"].astype(int).astype(str) + "|" +
        pair_base["physical_run"].astype(str)
    )

    core_pair_metrics = [
        "start_APL_BOX", "finish_APL_BOX_100ns", "APL_change_BOX",
        "APL_percent_change_BOX", "APL_mean_last10ns_BOX",
        "APL_std_last10ns_BOX", "start_thickness",
        "finish_thickness_100ns", "thickness_change",
    ] + [c for c in merged.columns if c.startswith("morph__")]

    core_pair_metrics = list(dict.fromkeys(
        [c for c in core_pair_metrics if c in pair_base.columns]
    ))

    pair_records = []
    pair_wide_records = []

    for pair_key, g in pair_base.groupby("pair_key", dropna=False):
        gm = g[g["bilayer_type"] == MAMMALIAN]
        gb = g[g["bilayer_type"] == BACTERIAL]
        if len(gm) != 1 or len(gb) != 1:
            continue
        rm = gm.iloc[0]
        rb = gb.iloc[0]

        identity = {
            "pair_key": pair_key,
            "molecule": rb["molecule"],
            "molecule_type": rb["molecule_type"],
            "length": rb["length"],
            "nmol": rb["nmol"],
            "physical_run": rb["physical_run"],
            "manual_label_mammalian": rm.get("manual_label"),
            "manual_label_bacterial": rb.get("manual_label"),
            "annotation_status_mammalian": rm.get("annotation_status"),
            "annotation_status_bacterial": rb.get("annotation_status"),
            "evidence_mammalian": rm.get("annotation_evidence_kind"),
            "evidence_bacterial": rb.get("annotation_evidence_kind"),
            "pair_primary_eligible": bool(
                bool(rm.get("primary_system_eligible", False))
                and bool(rb.get("primary_system_eligible", False))
            ),
        }

        # Membrane rows.
        for order, row_type, r in [
            (1, "mammalian", rm),
            (2, "bacterial", rb),
        ]:
            rec = dict(identity)
            rec.update({
                "row_order": order,
                "row_type": row_type,
                "bilayer_type": r["bilayer_type"],
                "manual_label": r.get("manual_label"),
                "annotation_status": r.get("annotation_status"),
                "annotation_evidence_kind": r.get("annotation_evidence_kind"),
                "physical_key": r.get("physical_key"),
                "system_dir": r.get("system_dir"),
                "ddAPL": np.nan,
            })
            for c in core_pair_metrics:
                rec[c] = r.get(c)
            pair_records.append(rec)

        # Difference row. Only whitelisted same-meaning metrics are subtracted.
        diff = dict(identity)
        diff.update({
            "row_order": 3,
            "row_type": "bacterial_minus_mammalian",
            "bilayer_type": "POPE75POPG25_MINUS_POPC60CHOL40",
            "manual_label": np.nan,
            "annotation_status": "paired_difference",
            "annotation_evidence_kind": np.nan,
            "physical_key": np.nan,
            "system_dir": np.nan,
        })
        for c in core_pair_metrics:
            diff[c] = np.nan
        for c in PAIR_DIFFERENCE_COLS:
            if c in core_pair_metrics:
                bv = pd.to_numeric(pd.Series([rb.get(c)]), errors="coerce").iloc[0]
                mv = pd.to_numeric(pd.Series([rm.get(c)]), errors="coerce").iloc[0]
                if pd.notna(bv) and pd.notna(mv):
                    diff[c] = bv - mv

        b_dapl = pd.to_numeric(
            pd.Series([rb.get("APL_change_BOX")]), errors="coerce"
        ).iloc[0]
        m_dapl = pd.to_numeric(
            pd.Series([rm.get("APL_change_BOX")]), errors="coerce"
        ).iloc[0]
        diff["ddAPL"] = (
            b_dapl - m_dapl
            if pd.notna(b_dapl) and pd.notna(m_dapl)
            else np.nan
        )
        pair_records.append(diff)

        wide = dict(identity)
        wide.update({
            "physical_key_mammalian": rm.get("physical_key"),
            "physical_key_bacterial": rb.get("physical_key"),
            "dAPL_mammalian": m_dapl,
            "dAPL_bacterial": b_dapl,
            "ddAPL": diff["ddAPL"],
            "dAPL_sum": (
                b_dapl + m_dapl
                if pd.notna(b_dapl) and pd.notna(m_dapl) else np.nan
            ),
            "mean_dAPL": (
                (b_dapl + m_dapl) / 2
                if pd.notna(b_dapl) and pd.notna(m_dapl) else np.nan
            ),
            "min_dAPL_between_membranes": (
                min(b_dapl, m_dapl)
                if pd.notna(b_dapl) and pd.notna(m_dapl) else np.nan
            ),
            "max_dAPL_between_membranes": (
                max(b_dapl, m_dapl)
                if pd.notna(b_dapl) and pd.notna(m_dapl) else np.nan
            ),
            "abs_ddAPL": (
                abs(diff["ddAPL"]) if pd.notna(diff["ddAPL"]) else np.nan
            ),
        })
        for c in core_pair_metrics:
            wide[f"mammalian__{c}"] = rm.get(c)
            wide[f"bacterial__{c}"] = rb.get(c)
            if c in PAIR_DIFFERENCE_COLS:
                bv = pd.to_numeric(pd.Series([rb.get(c)]), errors="coerce").iloc[0]
                mv = pd.to_numeric(pd.Series([rm.get(c)]), errors="coerce").iloc[0]
                wide[f"bacterial_minus_mammalian__{c}"] = (
                    bv - mv if pd.notna(bv) and pd.notna(mv) else np.nan
                )
        pair_wide_records.append(wide)

    paired = pd.DataFrame(pair_records).sort_values(["pair_key", "row_order"])
    pair_wide = pd.DataFrame(pair_wide_records).sort_values("pair_key")

    paired.to_csv(outdir / "03_paired_three_row_all.csv", index=False)

    is_tri_pair = (
        (pair_wide["molecule_type"] == "peptoid")
        & (pd.to_numeric(pair_wide["length"], errors="coerce") == 3)
    )
    tri_pair_keys = set(pair_wide.loc[is_tri_pair, "pair_key"])
    tri_three = paired[paired["pair_key"].isin(tri_pair_keys)].copy()
    tri_three.to_csv(outdir / "04_tripeptoid_three_row_all.csv", index=False)

    primary_wide = pair_wide[
        is_tri_pair & pair_wide["pair_primary_eligible"].fillna(False)
    ].copy()
    primary_keys = set(primary_wide["pair_key"])
    primary_three = paired[paired["pair_key"].isin(primary_keys)].copy()
    primary_three.to_csv(
        outdir / "05_tripeptoid_three_row_primary.csv", index=False
    )
    primary_wide.to_csv(
        outdir / "06_tripeptoid_pairs_wide_primary.csv", index=False
    )

    # ------------------------------------------------------------------
    # 7. Initial descriptive analysis
    # ------------------------------------------------------------------
    primary_system_rows = primary_three[
        primary_three["row_type"].isin(["mammalian", "bacterial"])
    ].copy()

    label_counts = (
        primary_system_rows.groupby(
            ["row_type", "manual_label"], dropna=False
        ).size().rename("n").reset_index()
    )
    label_counts.to_csv(outdir / "14_label_counts.csv", index=False)

    pair_label_counts = (
        primary_wide.groupby(
            ["manual_label_mammalian", "manual_label_bacterial"],
            dropna=False,
        ).size().rename("n_pairs").reset_index()
        .sort_values("n_pairs", ascending=False)
    )
    pair_label_counts.to_csv(outdir / "15_pair_label_counts.csv", index=False)

    ddapl_by_pair = safe_numeric_summary(
        primary_wide,
        ["manual_label_mammalian", "manual_label_bacterial"],
        ["ddAPL", "dAPL_mammalian", "dAPL_bacterial"],
    )
    ddapl_by_pair.to_csv(outdir / "16_ddapl_by_label_pair.csv", index=False)

    label_metric_list = [
        "APL_change_BOX", "APL_percent_change_BOX", "thickness_change",
    ] + [
        c for c in core_pair_metrics
        if c.startswith("morph__")
        and (
            "endpoint__midplane_excess_area_fraction" in c
            or "endpoint__height_rms_nm" in c
            or "endpoint__H_rms_nm_inv" in c
            or "endpoint__thickness_median_nm" in c
            or "delta__midplane_excess_area_fraction" in c
            or "delta__height_rms_nm" in c
            or "delta__H_rms_nm_inv" in c
            or "delta__thickness_median_nm" in c
        )
    ]
    label_metric_summary = safe_numeric_summary(
        primary_system_rows,
        ["row_type", "manual_label"],
        label_metric_list,
    )
    label_metric_summary.to_csv(
        outdir / "17_metric_summary_by_label.csv", index=False
    )

    corr_parts = []
    system_corr_metrics = [
        "APL_change_BOX", "APL_percent_change_BOX", "thickness_change",
    ] + [c for c in label_metric_list if c.startswith("morph__")]
    if not primary_system_rows.empty:
        corr_parts.append(
            spearman_long(
                primary_system_rows,
                system_corr_metrics,
                "primary_tripeptoid_system_rows",
            )
        )

    wide_corr_metrics = [
        "dAPL_mammalian", "dAPL_bacterial", "ddAPL", "abs_ddAPL",
        "mean_dAPL", "min_dAPL_between_membranes",
        "max_dAPL_between_membranes",
    ] + [
        c for c in primary_wide.columns
        if c.startswith("bacterial_minus_mammalian__morph__")
    ]
    if not primary_wide.empty:
        corr_parts.append(
            spearman_long(
                primary_wide,
                wide_corr_metrics,
                "primary_tripeptoid_pairs",
            )
        )
    correlations = (
        pd.concat(corr_parts, ignore_index=True)
        if corr_parts else
        pd.DataFrame(columns=["dataset", "metric_a", "metric_b", "n", "spearman_rho"])
    )
    correlations.to_csv(outdir / "18_numeric_correlations.csv", index=False)

    target_candidates = [
        "dAPL_bacterial", "dAPL_mammalian", "ddAPL", "abs_ddAPL",
        "mean_dAPL", "min_dAPL_between_membranes",
        "max_dAPL_between_membranes",
    ]
    trows = []
    for metric in target_candidates:
        v = pd.to_numeric(primary_wide[metric], errors="coerce").dropna()
        if v.empty:
            continue
        trows.append({
            "candidate": metric,
            "n_pairs": int(v.count()),
            "mean": v.mean(),
            "std": v.std(ddof=1),
            "median": v.median(),
            "q25": v.quantile(0.25),
            "q75": v.quantile(0.75),
            "min": v.min(),
            "max": v.max(),
            "interpretation_note": {
                "dAPL_bacterial": "canonical projected ΔAPL in POPE75:POPG25",
                "dAPL_mammalian": "canonical projected ΔAPL in POPC60:CHOL40",
                "ddAPL": "ΔAPL_bacterial - ΔAPL_mammalian",
                "abs_ddAPL": "magnitude of membrane selectivity; loses direction",
                "mean_dAPL": "mean projected response across the two membranes",
                "min_dAPL_between_membranes": "lower of the two membrane responses; strong-both response descriptor",
                "max_dAPL_between_membranes": "higher of the two membrane responses",
            }[metric],
        })
    pd.DataFrame(trows).to_csv(
        outdir / "19_target_candidate_summary.csv", index=False
    )

    pd.DataFrame(pilot_coverage).to_csv(
        outdir / "20_pilot_source_coverage.csv", index=False
    )

    # Current explicit photo exceptions from the canonical export.
    unresolved = pd.read_csv(unresolved_path, low_memory=False)
    unresolved.to_csv(outdir / "21_photo_exceptions.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Support/source tables
    # ------------------------------------------------------------------
    if not morph_selected.empty:
        morph_selected.to_csv(outdir / "support_morphology_core.csv", index=False)
    if not lipid_system_selected.empty:
        lipid_system_selected.to_csv(
            outdir / "support_local_lipid_system.csv", index=False
        )
    if not lipid_leaflet_pivot.empty:
        lipid_leaflet_pivot.to_csv(
            outdir / "support_local_lipid_leaflets_pivoted.csv", index=False
        )
    if not smoke_selected.empty:
        smoke_selected.to_csv(
            outdir / "support_explicit_leaflet_smoke.csv", index=False
        )
    if not dense_qc_selected.empty:
        dense_qc_selected.to_csv(
            outdir / "support_dense_chol_qc.csv", index=False
        )

    controls_path = bilayer_output_dir / "bilayer_only_APL_BOX_summary.csv"
    if controls_path.exists():
        controls = pd.read_csv(controls_path, low_memory=False)
        controls.to_csv(outdir / "support_bilayer_only_controls.csv", index=False)
        source_hashes[str(controls_path)] = sha256(controls_path)

    if pilot_files["metric_qc"].exists():
        pd.read_csv(pilot_files["metric_qc"]).to_csv(
            outdir / "support_morphology_metric_qc.csv", index=False
        )
        source_hashes[str(pilot_files["metric_qc"])] = sha256(
            pilot_files["metric_qc"]
        )
    if pilot_files["smoothing"].exists():
        pd.read_csv(pilot_files["smoothing"]).to_csv(
            outdir / "support_morphology_smoothing_stability.csv", index=False
        )
        source_hashes[str(pilot_files["smoothing"])] = sha256(
            pilot_files["smoothing"]
        )

    # ------------------------------------------------------------------
    # 9. Metric dictionary
    # ------------------------------------------------------------------
    md = []
    add_metric_dictionary(
        md, "APL_change_BOX", "functional response", "A^2 per membrane molecule",
        "masterdataanalysis.csv", "canonical",
        True, "ΔAPL = finish_APL_BOX_100ns - start_APL_BOX"
    )
    add_metric_dictionary(
        md, "ddAPL", "paired functional response", "A^2 per membrane molecule",
        "derived paired table", "canonical derived",
        False, "ddAPL = ΔAPL_bacterial - ΔAPL_mammalian; only populated on difference row"
    )
    add_metric_dictionary(
        md, "APL_percent_change_BOX", "functional response", "%",
        "masterdataanalysis.csv", "canonical",
        True, "Percent change under canonical BOX convention"
    )
    add_metric_dictionary(
        md, "thickness_change", "structural response", "A",
        "masterdataanalysis.csv", "existing analysis",
        True, "Source units preserved from masterdataanalysis.csv"
    )
    for src in MORPH_CORE_SOURCE_COLS:
        metric = "morph__" + src
        add_metric_dictionary(
            md, metric, "morphology", metric_units(metric),
            "pfp_morphology_pilot grid32/4nm",
            "validated/core pilot" if any(
                x in src for x in [
                    "midplane_excess_area_fraction", "height_rms_nm",
                    "H_rms_nm_inv", "thickness_median_nm"
                ]
            ) else "morphology QC",
            metric in PAIR_DIFFERENCE_COLS,
            "PO4 surface marker is geometry only; canonical APL remains BOX-based"
        )
    for c in merged.columns:
        if c.startswith("lipid_system__"):
            add_metric_dictionary(
                md, c, "local lipid state", metric_units(c),
                "pfp_morphology_pilot lipid_state_v1",
                "pilot; not production; not geodesic density",
                False,
                "Do not infer causality; CHOL interpretation requires leaflet QC context"
            )
        elif c.startswith("lipid_upper__") or c.startswith("lipid_lower__"):
            add_metric_dictionary(
                md, c, "leaflet local lipid state", metric_units(c),
                "pfp_morphology_pilot lipid_state_v1",
                "pilot; not independent replicate",
                False,
                "Upper/lower are paired leaflets of one simulation"
            )
        elif c.startswith("leaflet_smoke__"):
            add_metric_dictionary(
                md, c, "explicit leaflet geometry/QC", metric_units(c),
                "pfp_morphology_pilot morphology_leaflets/smoke",
                "small smoke pilot",
                False,
                "Upper/lower are paired; QC flags retained"
            )
        elif c.startswith("dense_chol_qc__"):
            add_metric_dictionary(
                md, c, "CHOL assignment QC", metric_units(c),
                "pfp_morphology_pilot morphology_leaflets/chol_dense_1ns",
                "dense QC pilot",
                False,
                "Overlaps smoke pilot; not an independent replicate"
            )
    pd.DataFrame(md).drop_duplicates("metric").to_csv(
        outdir / "08_metric_dictionary.csv", index=False
    )

    # ------------------------------------------------------------------
    # 10. Coverage and validation
    # ------------------------------------------------------------------
    coverage = pd.DataFrame([
        {"item": "canonical_annotation_rows", "n": len(ann)},
        {"item": "matchable_unambiguous_annotation_rows", "n": len(ann_matchable)},
        {"item": "concentration_ambiguous_annotation_rows", "n": len(ambiguities)},
        {"item": "simulation_master_rows", "n": len(sim)},
        {"item": "system_master_rows", "n": len(merged)},
        {"item": "annotation_and_simulation_rows", "n": int((merged["join_status"] == "annotation_and_simulation").sum())},
        {"item": "annotation_only_rows", "n": int((merged["join_status"] == "annotation_only").sum())},
        {"item": "simulation_only_rows", "n": int((merged["join_status"] == "simulation_only").sum())},
        {"item": "annotation_concentration_ambiguous_rows", "n": int((merged["join_status"] == "annotation_concentration_ambiguous").sum())},
        {"item": "all_complete_bacterial_mammalian_pairs", "n": len(pair_wide)},
        {"item": "all_tripeptoid_pairs", "n": int(is_tri_pair.sum())},
        {"item": "primary_tripeptoid_pairs", "n": len(primary_wide)},
        {"item": "primary_tripeptoid_three_row_rows", "n": len(primary_three)},
        {"item": "POPC80POPS20_system_rows", "n": int((merged["bilayer_type"] == POPS).sum())},
    ])
    coverage.to_csv(outdir / "09_join_coverage.csv", index=False)

    # Validate all three-row groups.
    primary_group_sizes = primary_three.groupby("pair_key").size()
    three_row_ok = bool(
        len(primary_group_sizes) == len(primary_wide)
        and (primary_group_sizes == 3).all()
    )
    diff_rows = primary_three[
        primary_three["row_type"] == "bacterial_minus_mammalian"
    ]
    diff_label_blank = bool(diff_rows["manual_label"].isna().all())
    ddapl_recalc = primary_wide["dAPL_bacterial"] - primary_wide["dAPL_mammalian"]
    ddapl_max_error = (
        float((primary_wide["ddAPL"] - ddapl_recalc).abs().max())
        if len(primary_wide) else np.nan
    )
    apl_max_error = (
        float(sim["_apl_error"].abs().max())
        if sim["_apl_error"].notna().any() else np.nan
    )

    # Check primary rows really are labelled STACKED on both membranes.
    primary_evidence_ok = bool(
        (primary_wide["evidence_mammalian"] == "STACKED").all()
        and (primary_wide["evidence_bacterial"] == "STACKED").all()
        and (primary_wide["annotation_status_mammalian"] == "labelled").all()
        and (primary_wide["annotation_status_bacterial"] == "labelled").all()
    )

    validation = {
        "status": "PASS",
        "read_only_sources": True,
        "canonical_annotation_rows": int(len(ann)),
        "canonical_export_summary": json.loads(canon_summary_path.read_text()),
        "simulation_master_rows": int(len(sim)),
        "system_master_rows": int(len(merged)),
        "canonical_box_dAPL_max_abs_recalc_error": apl_max_error,
        "master_run_metadata_path_mismatches": int(
            sim["run_metadata_path_mismatch"].sum()
        ),
        "concentration_ambiguous_annotation_rows": int(len(ambiguities)),
        "complete_bacterial_mammalian_pairs": int(len(pair_wide)),
        "tripeptoid_pairs": int(is_tri_pair.sum()),
        "primary_tripeptoid_pairs": int(len(primary_wide)),
        "primary_tripeptoid_three_row_rows": int(len(primary_three)),
        "primary_groups_exactly_three_rows": three_row_ok,
        "difference_rows_have_blank_single_manual_label": diff_label_blank,
        "primary_pairs_both_labelled_stacked": primary_evidence_ok,
        "ddAPL_max_abs_recalc_error": ddapl_max_error,
        "pilot_coverage": pilot_coverage,
        "source_sha256": source_hashes,
        "definitions": {
            "canonical_dAPL": "finish_APL_BOX_100ns - start_APL_BOX",
            "ddAPL": "APL_change_BOX_bacterial - APL_change_BOX_mammalian",
            "difference_row": "bacterial minus mammalian only for whitelisted same-meaning numeric metrics",
            "primary_tripeptoid_pair": (
                "peptoid length 3; both POPC60CHOL40 and POPE75POPG25 "
                "physical rows present; both manual labels labelled with STACKED "
                "evidence; both simulations complete to canonical 100 ns analysis"
            ),
        },
    }

    if not (
        three_row_ok
        and diff_label_blank
        and primary_evidence_ok
        and (pd.isna(ddapl_max_error) or ddapl_max_error < 1e-10)
        and (pd.isna(apl_max_error) or apl_max_error < 1e-8)
    ):
        validation["status"] = "REVIEW_REQUIRED"

    (outdir / "VALIDATION.json").write_text(json.dumps(validation, indent=2))

    readme = f"""# PFPeptoids canonical manual + simulation master analysis

## Main result

This directory was generated from the canonical annotation export and
`analysis/masterdataanalysis.csv` without altering either source.

The main requested three-row representation is:

1. `mammalian` = `{MAMMALIAN}`
2. `bacterial` = `{BACTERIAL}`
3. `bacterial_minus_mammalian`

`ddAPL = ΔAPL_bacterial - ΔAPL_mammalian`.

The third row does **not** subtract categorical labels or species-specific
local-lipid measurements. It contains the two source labels in
`manual_label_mammalian` and `manual_label_bacterial`; numeric subtraction is
restricted to metrics listed as allowed in `08_metric_dictionary.csv`.

## Primary dataset

`05_tripeptoid_three_row_primary.csv` is the first analysis dataset.

A primary pair requires:
- peptoid length = 3;
- both mammalian and bacterial physical simulations;
- canonical 100 ns simulation analysis complete;
- both manual annotations `labelled`;
- both supported by `STACKED` evidence.

`{POPS}` is retained separately in `07_popc80pops20_systems.csv` and is never
silently pooled with either member of the main pair.

## Morphology

The main morphology columns use grid 32 / 4 nm smoothing from the validated
pilot. PO4 is a geometry marker only. Canonical APL remains BOX XY projected
area per membrane molecule.

Redundant morphology measures are not promoted into the core three-row table.
The retained core is height RMS, RMS mean curvature, midplane excess area and
median local-normal thickness, with baseline/endpoint/delta, temporal summaries
and QC where available.

## Local lipid state

Local-lipid outputs are retained in `01_system_master_extended.csv` and support
tables, but are explicitly marked as pilot/non-production. Leaflets are paired
parts of the same simulation, not independent observations.

CHOL near-midplane and assignment-change QC is preserved. Dense CHOL sampling
overlaps the smoke pilot and is not counted as an additional replicate.

## Annotation concentration ambiguity

Any annotation record containing evidence from more than one `nmol` is never
silently copied onto individual simulation concentrations. Such rows are
isolated in `10_annotation_concentration_ambiguities.csv`.

## Files to use first

- `02_system_master_core.csv`
- `05_tripeptoid_three_row_primary.csv`
- `06_tripeptoid_pairs_wide_primary.csv`
- `14_label_counts.csv`
- `15_pair_label_counts.csv`
- `16_ddapl_by_label_pair.csv`
- `17_metric_summary_by_label.csv`
- `18_numeric_correlations.csv`
- `19_target_candidate_summary.csv`
- `VALIDATION.json`

No composite score or final ML target is chosen by this script.
"""
    (outdir / "README.md").write_text(readme)

    # ZIP output without deleting the directory.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(outdir))

    print("=" * 72)
    print("PFPeptoids MASTER ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Validation: {validation['status']}")
    print(f"System master rows: {len(merged)}")
    print(f"Complete bacterial/mammalian pairs: {len(pair_wide)}")
    print(f"Tripeptoid pairs: {int(is_tri_pair.sum())}")
    print(f"Primary tripeptoid pairs: {len(primary_wide)}")
    print(f"Primary three-row rows: {len(primary_three)}")
    print(f"Concentration-ambiguous annotation rows: {len(ambiguities)}")
    print(f"Run metadata/path mismatches: {int(sim['run_metadata_path_mismatch'].sum())}")
    print()
    print(f"OUTPUT DIR: {outdir}")
    print(f"UPLOAD THIS ZIP: {zip_path}")

if __name__ == "__main__":
    main()
