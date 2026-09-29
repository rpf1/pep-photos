#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PFPeptoidsAnalysisSuite.py

Version: audit_apl_execution_0.3

Purpose
-------
Unified PFPeptoids analysis suite foundation with real APL execution.

Current implemented modules:
    1. Audit/discovery
    2. GROMACS Pull/Relax APL analysis
    3. APL summary CSV
    4. Per-system APL plots/data
    5. Binned APL + binned APL slope

Still audit-only placeholders:
    1. SASA
    2. Photo/VMD generation
    3. Collage generation
    4. Migration of old outputs

Save location:
    /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/PFPeptoidsAnalysisSuite.py

Recommended first run:
    RUN_ANALYSIS_AFTER_AUDIT = False

Then inspect:
    analysis/results/tables/audit_actions_needed.csv

To run APL:
    RUN_ANALYSIS_AFTER_AUDIT = True
    Type YES when prompted.

Command-line examples:
    python PFPeptoidsAnalysisSuite.py --audit
    python PFPeptoidsAnalysisSuite.py --audit --run-from-audit
    python PFPeptoidsAnalysisSuite.py --audit --run-from-audit --no-confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MDAnalysis as mda


# =============================================================================
# USER SETTINGS - SPYDER FRIENDLY
# =============================================================================

ANALYSIS_DIR = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis")
DATA_ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1")
RESULTS_ROOT = ANALYSIS_DIR / "results"

CPHMD_ROOTS: List[str] = []

TARGET_DIR_MODE = True

TARGET_DIRS = [
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1/POPC60CHOL40_W_WF/run_0",
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1/POPE75POPG25_W_WF/run_0",
]

RUN_AUDIT = True
FORCE_FULL_AUDIT = True
TRUST_PREVIOUS_STATE_JSON = False

# Safe default:
#   False = audit only
#   True  = audit, ask confirmation, then execute runnable modules
RUN_ANALYSIS_AFTER_AUDIT = True
REQUIRE_USER_CONFIRMATION_AFTER_AUDIT = True

# v0.3: APL can run. SASA/photos/collages are still audit-only placeholders.
RUN_APL = True
RUN_SASA = False
RUN_PHOTOS = False
RUN_COLLAGES = False
RUN_GROUPED_STACKS = True

RUN_GROMACS_PULL_RELAX = True
RUN_NAMD_CPHMD = True

FORCE_REANALYSE_APL = False
FORCE_REANALYSE_SASA = False
FORCE_REGENERATE_PHOTOS = False
FORCE_REGENERATE_COLLAGES = True

ALLOW_SASA_WITHOUT_BILAYER = True
SKIP_APL_IF_NO_BILAYER = True
SKIP_EMPTY_SASA_SELECTIONS = True
INCLUDE_INCOMPLETE_GROMACS_SYSTEMS_IN_AUDIT = True


# =============================================================================
# DISCOVERY EXCLUSION SETTINGS
# =============================================================================

EXCLUDED_DIR_NAMES = {
    "failures",
    "failed",
    "failure",
    "backup",
    "backups",
    "old",
    "archive",
    "archives",
    "tmp",
    "temp",
    "__pycache__",
}

EXCLUDED_DIR_NAME_CONTAINS = {
    "step 0 stall",
}


# =============================================================================
# APL SETTINGS
# =============================================================================

TARGET_RELAX_TIME_NS = 100.0
TARGET_RELAX_TOLERANCE_NS = 0.5

PO4_SELECTION = "name PO4"
PHOSPHOLIPID_RESNAMES = {"POPC", "POPS", "POPE", "POPG", "DPPC", "DTPC"}
CHOL_RESNAMES = {"CHOL"}

SYSTEM_TYPE_ALIASES = {
    "POPC60CHOL40_20Az": "POPC60CHOL40_W_WF",
    "POPE75POPG25_20Az": "POPE75POPG25_W_WF",
    "POPC80POPS20_20Az": "POPC80POPS20_W_WF",
}

SYSTEM_TYPE_TO_ASSUMED_PHOSPHOLIPID_FRACTION = {
    "POPC60CHOL40_W_WF": 0.60,
    "POPC80POPS20_W_WF": 1.00,
    "POPE75POPG25_W_WF": 1.00,
    "POPC60CHOL40_20Az": 0.60,
    "POPE75POPG25_20Az": 1.00,
    "POPC80POPS20_20Az": 1.00,
}

BILAYER_PRIORITY = [
    "POPC60CHOL40_20Az",
    "POPE75POPG25_20Az",
    "POPC80POPS20_20Az",
    "POPC60CHOL40_W_WF",
    "POPE75POPG25_W_WF",
    "POPC80POPS20_W_WF",
]

PLOT_CFG = {
    "font_family": "DejaVu Sans",
    "font_size": 12,
    "title_size": 14,
    "axis_label_size": 13,
    "tick_label_size": 11,
    "legend_size": 10,
    "line_width": 2.0,
    "figure_size": (8.0, 5.0),
    "dpi": 300,

    "pull_title": "Pull APL vs Time",
    "relax_title": "Relax APL vs Time",
    "combined_title": "Combined APL vs Time",

    "x_label": "Time (ns)",
    "y_label": "APL (Å²)",

    "pull_color": "orange",
    "relax_color": "blue",

    "pull_xlim": None,
    "relax_xlim": None,
    "combined_xlim": None,

    "pull_ylim": (40, 85),
    "relax_ylim": (40, 85),
    "combined_ylim": (40, 85),

    "tight_layout": True,
}

BINNED_APL_CFG = {
    "enabled": True,
    "metric": "APL",
    "bin_size_ns": 5.0,

    "start_ns": 0.0,
    "end_ns": 200.0,

    "use_bin_center_for_x": False,
    "drop_incomplete_last_bin": False,

    "plot_enabled": True,
    "annotate_slope_values": True,
    "annotation_fmt": "{:.3f}",
    "annotation_rotation": 0,

    "apl_xlabel": "Time (ns)",
    "apl_ylabel": "Average APL (Å²)",
    "slope_xlabel": "Time (ns)",
    "slope_ylabel": "ΔAPL per bin",

    "apl_color": "black",
    "slope_color": "red",

    "pull_color": "orange",
    "relax_color": "blue",

    "show_grid": True,
    "show_apl_markers": True,
    "show_slope_markers": True,
    "apl_marker_size": 5.0,
    "slope_marker_size": 5.0,
    "annotation_font_size": 9,

    "show_phase_boundary": True,
    "phase_boundary_linestyle": "--",
    "phase_boundary_color": "black",
    "phase_boundary_linewidth": 1.0,

    "show_legend": True,

    "apl_xlim": (0, 200),
    "apl_ylim": (40, 85),

    "slope_xlim": (0, 200),

    "slope_mode": "delta_per_bin",
    "slope_ylim_fixed": (-2.0, 2.0),
    "slope_major_tick": 0.5,
    "slope_minor_tick": 0.25,
}


# =============================================================================
# SASA SETTINGS - AUDIT ONLY IN v0.3
# =============================================================================

SASA_TARGETS = [
    "polymer_all",
    "bilayer",
]

SASA_FRAME_MODE = "start_end"
SASA_STRIDE = 10
SASA_PROBE_RADIUS = 1.4


# =============================================================================
# PHOTO SETTINGS - AUDIT ONLY IN v0.3
# =============================================================================

PHOTO_MODE = "auto"
VMD_COMMAND = "vglrun vmd -e Photograph.tcl"
TCL_TEMPLATE = ANALYSIS_DIR / "Photograph.tcl"

TRIM_IMAGES = True
TRIM_FUZZ = "7%"

DRAW_LABELS = True
LABEL_FONT_SIZE = 28
TITLE_FONT_SIZE = 34
LABEL_COLOR = "black"
TITLE_COLOR = "black"
LABEL_PADDING = 12
TITLE_BAND_HEIGHT = 56
PANEL_LABEL_BAND_HEIGHT = 42

SHIFT_BILAYER_VIEW = True
BILAYER_TARGET_Z_FRACTION = 0.25
SHIFTED_OUTPUT_TAG = "bilayer_z025"


# =============================================================================
# OUTPUT FILES
# =============================================================================

TABLES_DIR = RESULTS_ROOT / "tables"
LOGS_DIR = RESULTS_ROOT / "logs"
SYSTEMS_DIR = RESULTS_ROOT / "systems"
GROUPED_DIR = RESULTS_ROOT / "grouped"

AUDIT_SYSTEMS_CSV = TABLES_DIR / "audit_systems.csv"
AUDIT_INPUTS_CSV = TABLES_DIR / "audit_inputs.csv"
AUDIT_OUTPUTS_CSV = TABLES_DIR / "audit_existing_outputs.csv"
AUDIT_ACTIONS_CSV = TABLES_DIR / "audit_actions_needed.csv"
AUDIT_SUMMARY_JSON = LOGS_DIR / "audit_report.json"
ANALYSIS_STATE_JSON = LOGS_DIR / "analysis_state.json"

APL_SUMMARY_CSV = TABLES_DIR / "apl_summary.csv"
APL_FAILED_CSV = TABLES_DIR / "apl_failed.csv"

SCRIPT_VERSION = "audit_apl_execution_0.3"


# =============================================================================
# CENTRAL RESIDUE DEFINITIONS
# =============================================================================

BILAYER_RESNAMES: Set[str] = {
    "POPC", "POPS", "POPE", "POPG", "DPPC", "DTPC", "CHOL",
}

WATER_RESNAMES: Set[str] = {
    "H2O", "W", "SW", "TW", "WF",
}

ION_RESNAMES: Set[str] = {
    "NA", "CL", "NA+", "CL-", "ION",
}

PEPTIDE_RESNAMES: Set[str] = {
    "ALAC", "ALAN", "ALA",
    "VALC", "VALN", "VAL",
    "ILEC", "ILEN", "ILE",
    "LEUC", "LEUN", "LEU",
    "METC", "METN", "MET",
    "CYSC", "CYSN", "CYS",

    "PROC", "PRON", "PRO",

    "SERC", "SERN", "SER",
    "THRC", "THRN", "THR",
    "ASNC", "ASNN", "ASN",
    "GLNC", "GLNN", "GLN",
    "HSEC", "HSEN", "HSE",
    "HIS", "HISC", "HISN",

    "ASPC", "ASPN", "ASP",
    "GLUC", "GLUN", "GLU",

    "PHEC", "PHEN", "PHE",
    "TYRC", "TYRN", "TYR",
    "TRPC", "TRPN", "TRP",

    "LYSC", "LYSN", "LYS",
    "ARGC", "ARGN", "ARG",
}

PEPTOID_RESNAMES: Set[str] = {
    "Na", "Nab", "Ni", "Nl", "Nm", "Nv",
    "Nn", "Nq", "Ns", "Nse", "Nt", "NmO",
    "Nd", "Ne",
    "Nf", "Nfn", "Nfe", "Nfex", "Nw", "Nwe", "Ny",
    "Nfe4Cl", "Nfe4Br", "Nfnaph",
    "Nfe[4Cl]", "Nfe[4Br]", "Nf[naph]",
    "Nk", "Nke", "Nr",
}

POLYMER_RESNAMES: Set[str] = PEPTIDE_RESNAMES | PEPTOID_RESNAMES
SOLVENT_RESNAMES: Set[str] = WATER_RESNAMES | ION_RESNAMES
NON_POLYMER_RESNAMES: Set[str] = BILAYER_RESNAMES | WATER_RESNAMES | ION_RESNAMES

VALID_SASA_TARGETS = {
    "bilayer",
    "peptide",
    "peptoid",
    "polymer_all",
    "whole_system",
}


# =============================================================================
# EXPECTED OUTPUT NAMES
# =============================================================================

APL_OUTPUTS = [
    "data/apl_timeseries.csv",
    "data/binned_apl.csv",
    "data/binned_apl_slope.csv",
    "plots/apl_pull.png",
    "plots/apl_relax.png",
    "plots/apl_combined.png",
    "plots/binned_apl.png",
    "plots/binned_apl_slope.png",
]

PHOTO_OUTPUTS_BY_MODE = {
    "bilayer": [
        "photos/snapshot_top_bilayer.png",
        "photos/snapshot_side_bilayer.png",
        "photos/snapshot_top.png",
        "photos/snapshot_side.png",
        "photos/snapshot_bottom.png",
        "photos/snapshot_water.png",
        "photos/photo_panel.png",
        "photos/vmd_render.tcl",
    ],
    "waterbox": [
        "photos/snapshot_top.png",
        "photos/snapshot_side.png",
        "photos/snapshot_bottom.png",
        "photos/snapshot_water.png",
        "photos/photo_panel.png",
        "photos/vmd_render.tcl",
    ],
    "polymer_only": [
        "photos/snapshot_top.png",
        "photos/snapshot_side.png",
        "photos/snapshot_bottom.png",
        "photos/photo_panel.png",
        "photos/vmd_render.tcl",
    ],
}

COLLAGE_SETS = [
    ["APL"],
    ["SASA"],
    ["PHOTO"],
    ["APL", "SASA"],
    ["PHOTO", "APL"],
    ["PHOTO", "SASA"],
    ["PHOTO", "APL", "SASA"],
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SystemTarget:
    system_id: str
    simulation_family: str

    system_dir: str
    molecule_name: str
    nmol: Optional[int]
    bilayer: str
    run_name: str
    stage_label: str

    pull_gro: str = ""
    pull_xtc: str = ""
    relax_gro: str = ""
    relax_xtc: str = ""
    pull_cpt: str = ""
    relax_cpt: str = ""

    namd_psf: str = ""
    namd_pdb: str = ""
    namd_dcd: str = ""
    namd_xst: str = ""
    namd_log: str = ""

    representative_structure: str = ""
    representative_trajectory: str = ""

    output_dir: str = ""

    has_pull: bool = False
    has_relax: bool = False
    has_namd: bool = False

    has_bilayer: bool = False
    has_peptide: bool = False
    has_peptoid: bool = False
    has_polymer: bool = False
    has_water: bool = False
    has_ions: bool = False

    detected_resnames: str = ""

    complete_for_apl: bool = False
    complete_for_sasa: bool = False
    complete_for_photos: bool = False

    discovery_status: str = "discovered"
    discovery_reason: str = ""


@dataclass
class AuditAction:
    system_id: str
    simulation_family: str
    molecule_name: str
    nmol: Optional[int]
    bilayer: str
    run_name: str
    analysis_type: str
    target: str
    action: str
    reason: str
    system_dir: str
    output_dir: str
    required_inputs: str
    missing_inputs: str
    expected_outputs: str
    missing_outputs: str


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def log_kv(key: str, value) -> None:
    print(f"{key:<36} {value}")


def mkdirs() -> None:
    for d in [RESULTS_ROOT, TABLES_DIR, LOGS_DIR, SYSTEMS_DIR, GROUPED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def clean_paths(paths: List[str]) -> List[str]:
    cleaned: List[str] = []

    for p in paths:
        if p is None:
            continue

        p_str = str(p).strip()

        if not p_str:
            continue

        if p_str.lower() in {"nan", "none", "null"}:
            continue

        cleaned.append(p_str)

    return cleaned


def path_exists(path_str: str) -> bool:
    if path_str is None:
        return False

    path_str = str(path_str).strip()

    if not path_str or path_str.lower() in {"nan", "none", "null"}:
        return False

    return Path(path_str).exists()


def file_size(path_str: str) -> int:
    if not path_exists(path_str):
        return 0
    try:
        return Path(path_str).stat().st_size
    except Exception:
        return 0


def newest_mtime(paths: List[str]) -> float:
    mtimes = []
    for p in clean_paths(paths):
        if path_exists(p):
            try:
                mtimes.append(Path(p).stat().st_mtime)
            except Exception:
                pass
    return max(mtimes) if mtimes else 0.0


def oldest_mtime(paths: List[str]) -> float:
    mtimes = []
    for p in clean_paths(paths):
        if path_exists(p):
            try:
                mtimes.append(Path(p).stat().st_mtime)
            except Exception:
                pass
    return min(mtimes) if mtimes else 0.0


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def path_is_excluded(path: Path) -> bool:
    parts_lower = [p.lower() for p in path.parts]

    for bad in EXCLUDED_DIR_NAMES:
        if bad.lower() in parts_lower:
            return True

    path_text = str(path).lower()
    for bad_text in EXCLUDED_DIR_NAME_CONTAINS:
        if bad_text.lower() in path_text:
            return True

    return False


def parse_system_components(folder_name: str) -> List[Dict[str, object]]:
    components = []

    for part in str(folder_name).split("|"):
        part = part.strip()
        if "_" not in part:
            raise ValueError(f"Component is missing '_count': {part}")

        name, count_str = part.rsplit("_", 1)
        name = name.strip()
        count = int(count_str)

        if not name:
            raise ValueError(f"Empty component name in: {part}")

        components.append({"name": name, "count": count})

    return components


def split_system_folder_name(folder_name: str) -> Tuple[str, Optional[int]]:
    try:
        components = parse_system_components(folder_name)
        nmol = sum(int(c["count"]) for c in components)
        return str(folder_name), nmol
    except Exception:
        if "_" in str(folder_name):
            name, possible_nmol = str(folder_name).rsplit("_", 1)
            return name, safe_int(possible_nmol)
        return str(folder_name), None


def parse_run_name_from_path(system_dir: Path) -> str:
    parent = system_dir.parent.name
    if parent.startswith("run_"):
        return parent
    return parent


def parse_bilayer_from_path(system_dir: Path) -> str:
    try:
        if system_dir.parent.name.startswith("run_"):
            return system_dir.parents[1].name
    except Exception:
        pass

    return "unknown_or_waterbox"


def output_dir_for_target(
    molecule_name: str,
    bilayer: str,
    run_name: str,
    simulation_family: str,
) -> Path:
    return (
        SYSTEMS_DIR
        / sanitize_filename(molecule_name)
        / sanitize_filename(bilayer)
        / sanitize_filename(run_name)
        / sanitize_filename(simulation_family)
    )


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    try:
        df.to_excel(path.with_suffix(".xlsx"), index=False)
    except Exception as exc:
        print(f"WARNING: could not write Excel copy for {path}: {exc}")


def append_dataframe_to_csv_dedup(
    new_df: pd.DataFrame,
    path: Path,
    dedup_subset: List[str],
) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            old_df = pd.read_csv(path)
        except Exception:
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()

    if old_df.empty:
        combined = new_df.copy()
    elif new_df.empty:
        combined = old_df.copy()
    else:
        combined = pd.concat([old_df, new_df], ignore_index=True)

    if not combined.empty and all(c in combined.columns for c in dedup_subset):
        combined = combined.drop_duplicates(subset=dedup_subset, keep="last")

    write_dataframe(combined, path)
    return combined


def wrap_title(title: str, width: int = 70) -> str:
    return "\n".join(textwrap.wrap(str(title), width=width, break_long_words=False))


def canonical_system_type(system_type: str) -> str:
    return SYSTEM_TYPE_ALIASES.get(system_type, system_type)


def pretty_bilayer_name(system_type: str) -> str:
    lookup = canonical_system_type(system_type)

    mapping = {
        "POPC60CHOL40_W_WF": "POPC60:CHOL40",
        "POPE75POPG25_W_WF": "POPE75:POPG25",
        "POPC80POPS20_W_WF": "POPC80:POPS20",
    }

    return mapping.get(lookup, str(system_type).replace("_W_WF", ""))


def ordered_bilayers(bilayers: List[str]) -> List[str]:
    ordered = [b for b in BILAYER_PRIORITY if b in bilayers]
    ordered.extend(sorted([b for b in bilayers if b not in BILAYER_PRIORITY]))
    return ordered


# =============================================================================
# SELECTION HELPERS
# =============================================================================

def mdaselect_resnames(resnames: Set[str]) -> str:
    clean = sorted(str(x) for x in resnames if str(x).strip())
    if not clean:
        raise ValueError("Cannot build selection from empty residue-name set.")
    return "resname " + " ".join(clean)


def build_sasa_selection(target_name: str) -> str:
    target_name = target_name.lower().strip()

    if target_name == "bilayer":
        return mdaselect_resnames(BILAYER_RESNAMES)
    if target_name == "peptide":
        return mdaselect_resnames(PEPTIDE_RESNAMES)
    if target_name == "peptoid":
        return mdaselect_resnames(PEPTOID_RESNAMES)
    if target_name == "polymer_all":
        return mdaselect_resnames(POLYMER_RESNAMES)
    if target_name == "whole_system":
        return "all"

    raise ValueError(f"Unknown SASA target: {target_name}. Allowed: {sorted(VALID_SASA_TARGETS)}")


# =============================================================================
# STRUCTURE PARSING FOR AUDIT
# =============================================================================

def parse_gro_resnames(gro_path: Path) -> Set[str]:
    resnames: Set[str] = set()

    if gro_path is None or not gro_path.exists():
        return resnames

    try:
        with open(gro_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return resnames

    if len(lines) < 3:
        return resnames

    for line in lines[2:-1]:
        if len(line) >= 10:
            rn = line[5:10].strip()
            if rn:
                resnames.add(rn)

    return resnames


def parse_pdb_resnames(pdb_path: Path) -> Set[str]:
    resnames: Set[str] = set()

    if pdb_path is None or not pdb_path.exists():
        return resnames

    try:
        with open(pdb_path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    rn = line[17:22].strip()
                    if rn:
                        resnames.add(rn)
    except Exception:
        return resnames

    return resnames


def classify_resnames(resnames: Set[str]) -> Dict[str, object]:
    has_bilayer = len(resnames & BILAYER_RESNAMES) > 0
    has_peptide = len(resnames & PEPTIDE_RESNAMES) > 0
    has_peptoid = len(resnames & PEPTOID_RESNAMES) > 0
    has_water = len(resnames & WATER_RESNAMES) > 0
    has_ions = len(resnames & ION_RESNAMES) > 0
    has_polymer = has_peptide or has_peptoid

    return {
        "has_bilayer": has_bilayer,
        "has_peptide": has_peptide,
        "has_peptoid": has_peptoid,
        "has_polymer": has_polymer,
        "has_water": has_water,
        "has_ions": has_ions,
        "detected_resnames": "|".join(sorted(resnames)),
    }


def representative_structure_for_gromacs(system_dir: Path) -> Optional[Path]:
    candidates = [
        system_dir / "Relax.gro",
        system_dir / "Pull.gro",
        system_dir / "Min.gro",
        system_dir / "min.gro",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def representative_structure_for_namd(system_dir: Path) -> Optional[Path]:
    pdbs = sorted(system_dir.glob("*.pdb"))
    if pdbs:
        preferred_terms = ["final", "relax", "cphmd", "system", "built", "start"]
        for term in preferred_terms:
            matches = [p for p in pdbs if term.lower() in p.name.lower()]
            if matches:
                return matches[0]
        return pdbs[0]
    return None


# =============================================================================
# GROMACS DISCOVERY
# =============================================================================

GROMACS_EXPECTED_FILES = [
    "Pull.gro",
    "Pull.xtc",
    "Relax.gro",
    "Relax.xtc",
]


def has_any_gromacs_marker(system_dir: Path) -> bool:
    names = {
        "Pull.gro", "Pull.xtc", "Relax.gro", "Relax.xtc",
        "pull.gro", "pull.xtc", "relax.gro", "relax.xtc",
    }
    try:
        existing = {p.name for p in system_dir.iterdir() if p.is_file()}
    except Exception:
        return False
    return bool(existing & names)


def has_all_gromacs_required(system_dir: Path) -> bool:
    return all((system_dir / name).exists() for name in GROMACS_EXPECTED_FILES)


def is_probable_gromacs_system_folder(system_dir: Path) -> bool:
    if not system_dir.is_dir():
        return False

    if path_is_excluded(system_dir):
        return False

    if not has_any_gromacs_marker(system_dir):
        return False

    return True


def scan_gromacs_target_directory(target_dir: Path) -> List[Path]:
    if not target_dir.exists():
        print(f"WARNING: target does not exist: {target_dir}")
        return []

    if not target_dir.is_dir():
        print(f"WARNING: target is not a directory: {target_dir}")
        return []

    if path_is_excluded(target_dir):
        return []

    if is_probable_gromacs_system_folder(target_dir):
        return [target_dir]

    folders = set()

    for marker in ["Pull.gro", "Pull.xtc", "Relax.gro", "Relax.xtc"]:
        for p in target_dir.rglob(marker):
            if path_is_excluded(p):
                continue

            if p.is_file():
                folder = p.parent

                if path_is_excluded(folder):
                    continue

                if is_probable_gromacs_system_folder(folder):
                    folders.add(folder)

    return sorted(folders)


def discover_gromacs_targets() -> List[SystemTarget]:
    folders: List[Path] = []

    if TARGET_DIR_MODE:
        for target in TARGET_DIRS:
            found = scan_gromacs_target_directory(Path(target))
            if not found:
                print(f"No probable GROMACS systems found under: {target}")
            folders.extend(found)
    else:
        folders = scan_gromacs_target_directory(DATA_ROOT)

    unique_folders = sorted(set(folders))
    targets: List[SystemTarget] = []

    for system_dir in unique_folders:
        molecule_name, nmol = split_system_folder_name(system_dir.name)
        run_name = parse_run_name_from_path(system_dir)
        bilayer = parse_bilayer_from_path(system_dir)

        pull_gro = system_dir / "Pull.gro"
        pull_xtc = system_dir / "Pull.xtc"
        relax_gro = system_dir / "Relax.gro"
        relax_xtc = system_dir / "Relax.xtc"
        pull_cpt = system_dir / "Pull.cpt"
        relax_cpt = system_dir / "Relax.cpt"

        has_pull = pull_gro.exists() and pull_xtc.exists()
        has_relax = relax_gro.exists() and relax_xtc.exists()

        rep_structure = representative_structure_for_gromacs(system_dir)
        rep_traj = relax_xtc if relax_xtc.exists() else pull_xtc if pull_xtc.exists() else None

        resnames = parse_gro_resnames(rep_structure) if rep_structure is not None else set()
        cls = classify_resnames(resnames)

        output_dir = output_dir_for_target(
            molecule_name=molecule_name,
            bilayer=bilayer,
            run_name=run_name,
            simulation_family="gromacs_pull_relax",
        )

        complete_all = has_all_gromacs_required(system_dir)

        if complete_all:
            discovery_status = "complete_input_set"
            discovery_reason = "Pull.gro, Pull.xtc, Relax.gro, Relax.xtc found"
        else:
            missing = [x for x in GROMACS_EXPECTED_FILES if not (system_dir / x).exists()]
            discovery_status = "incomplete_input_set"
            discovery_reason = "Missing: " + ", ".join(missing)

        system_id = stable_hash(f"gromacs|{system_dir}")

        complete_for_apl = complete_all and bool(cls["has_bilayer"])
        complete_for_sasa = (has_pull or has_relax) and bool(cls["has_polymer"] or cls["has_bilayer"])
        complete_for_photos = (has_pull or has_relax) and bool(cls["has_polymer"] or cls["has_bilayer"])

        target = SystemTarget(
            system_id=system_id,
            simulation_family="gromacs_pull_relax",
            system_dir=str(system_dir),
            molecule_name=molecule_name,
            nmol=nmol,
            bilayer=bilayer,
            run_name=run_name,
            stage_label="PullRelax",

            pull_gro=str(pull_gro) if pull_gro.exists() else "",
            pull_xtc=str(pull_xtc) if pull_xtc.exists() else "",
            relax_gro=str(relax_gro) if relax_gro.exists() else "",
            relax_xtc=str(relax_xtc) if relax_xtc.exists() else "",
            pull_cpt=str(pull_cpt) if pull_cpt.exists() else "",
            relax_cpt=str(relax_cpt) if relax_cpt.exists() else "",

            representative_structure=str(rep_structure) if rep_structure is not None else "",
            representative_trajectory=str(rep_traj) if rep_traj is not None and rep_traj.exists() else "",

            output_dir=str(output_dir),

            has_pull=has_pull,
            has_relax=has_relax,
            has_namd=False,

            has_bilayer=bool(cls["has_bilayer"]),
            has_peptide=bool(cls["has_peptide"]),
            has_peptoid=bool(cls["has_peptoid"]),
            has_polymer=bool(cls["has_polymer"]),
            has_water=bool(cls["has_water"]),
            has_ions=bool(cls["has_ions"]),
            detected_resnames=str(cls["detected_resnames"]),

            complete_for_apl=complete_for_apl,
            complete_for_sasa=complete_for_sasa,
            complete_for_photos=complete_for_photos,

            discovery_status=discovery_status,
            discovery_reason=discovery_reason,
        )

        if complete_all or INCLUDE_INCOMPLETE_GROMACS_SYSTEMS_IN_AUDIT:
            targets.append(target)

    return targets


# =============================================================================
# NAMD / CPHMD DISCOVERY - AUDIT FOUNDATION ONLY
# =============================================================================

def find_first_by_suffix(system_dir: Path, suffixes: List[str]) -> str:
    for suffix in suffixes:
        matches = sorted(system_dir.glob(f"*{suffix}"))
        if matches:
            return str(matches[0])
    return ""


def is_probable_namd_cphmd_folder(system_dir: Path) -> bool:
    if not system_dir.is_dir():
        return False

    if path_is_excluded(system_dir):
        return False

    try:
        files = [p for p in system_dir.iterdir() if p.is_file()]
    except Exception:
        return False

    names = [p.name.lower() for p in files]
    has_dcd = any(n.endswith(".dcd") for n in names)
    has_psf = any(n.endswith(".psf") for n in names)
    has_pdb = any(n.endswith(".pdb") for n in names)
    has_cphmd_name = any("cph" in n or "namd" in n for n in names)

    return has_dcd and (has_psf or has_pdb or has_cphmd_name)


def scan_namd_cphmd_root(root: Path) -> List[Path]:
    if not root.exists():
        print(f"WARNING: CpHMD root does not exist: {root}")
        return []

    if not root.is_dir():
        print(f"WARNING: CpHMD root is not a directory: {root}")
        return []

    if path_is_excluded(root):
        return []

    folders = set()

    if is_probable_namd_cphmd_folder(root):
        folders.add(root)

    for dcd in root.rglob("*.dcd"):
        if path_is_excluded(dcd):
            continue

        folder = dcd.parent

        if path_is_excluded(folder):
            continue

        if is_probable_namd_cphmd_folder(folder):
            folders.add(folder)

    return sorted(folders)


def discover_namd_cphmd_targets() -> List[SystemTarget]:
    targets: List[SystemTarget] = []

    for root_str in CPHMD_ROOTS:
        folders = scan_namd_cphmd_root(Path(root_str))

        for system_dir in folders:
            molecule_name, nmol = split_system_folder_name(system_dir.name)
            run_name = parse_run_name_from_path(system_dir)
            bilayer = parse_bilayer_from_path(system_dir)

            psf = find_first_by_suffix(system_dir, [".psf"])
            pdb = find_first_by_suffix(system_dir, [".pdb"])
            dcd = find_first_by_suffix(system_dir, [".dcd"])
            xst = find_first_by_suffix(system_dir, [".xst"])
            log = find_first_by_suffix(system_dir, [".log"])

            rep_structure = Path(pdb) if pdb else representative_structure_for_namd(system_dir)
            rep_traj = Path(dcd) if dcd else None

            resnames = set()
            if rep_structure is not None and Path(rep_structure).exists():
                resnames = parse_pdb_resnames(Path(rep_structure))

            cls = classify_resnames(resnames)

            output_dir = output_dir_for_target(
                molecule_name=molecule_name,
                bilayer=bilayer,
                run_name=run_name,
                simulation_family="namd_cphmd",
            )

            has_namd = bool(dcd) and (bool(psf) or bool(pdb))

            if has_namd:
                discovery_status = "complete_or_probable_cphmd_input_set"
                discovery_reason = "DCD plus PSF/PDB detected"
            else:
                discovery_status = "incomplete_cphmd_input_set"
                discovery_reason = "Missing DCD and/or PSF/PDB"

            target = SystemTarget(
                system_id=stable_hash(f"namd_cphmd|{system_dir}"),
                simulation_family="namd_cphmd",
                system_dir=str(system_dir),
                molecule_name=molecule_name,
                nmol=nmol,
                bilayer=bilayer,
                run_name=run_name,
                stage_label="CpHMD",

                namd_psf=psf,
                namd_pdb=pdb,
                namd_dcd=dcd,
                namd_xst=xst,
                namd_log=log,

                representative_structure=str(rep_structure) if rep_structure is not None else "",
                representative_trajectory=str(rep_traj) if rep_traj is not None else "",

                output_dir=str(output_dir),

                has_pull=False,
                has_relax=False,
                has_namd=has_namd,

                has_bilayer=bool(cls["has_bilayer"]),
                has_peptide=bool(cls["has_peptide"]),
                has_peptoid=bool(cls["has_peptoid"]),
                has_polymer=bool(cls["has_polymer"]),
                has_water=bool(cls["has_water"]),
                has_ions=bool(cls["has_ions"]),
                detected_resnames=str(cls["detected_resnames"]),

                complete_for_apl=bool(cls["has_bilayer"]) and has_namd,
                complete_for_sasa=has_namd and bool(cls["has_polymer"] or cls["has_bilayer"]),
                complete_for_photos=bool(pdb) and bool(cls["has_polymer"] or cls["has_bilayer"]),

                discovery_status=discovery_status,
                discovery_reason=discovery_reason,
            )

            targets.append(target)

    return targets


# =============================================================================
# OUTPUT EXPECTATION HELPERS
# =============================================================================

def expected_sasa_outputs_for_target(sasa_target: str) -> List[str]:
    return [
        f"data/sasa_{sasa_target}_timeseries.csv",
        f"plots/sasa_{sasa_target}_combined.png",
    ]


def expected_collage_outputs() -> List[str]:
    return ["collages/" + "_".join(panel_set) + ".png" for panel_set in COLLAGE_SETS]


def choose_photo_mode(target: SystemTarget) -> str:
    if PHOTO_MODE != "auto":
        return PHOTO_MODE

    if target.has_bilayer:
        return "bilayer"

    if target.has_polymer and target.has_water:
        return "waterbox"

    if target.has_polymer:
        return "polymer_only"

    return "unknown"


def output_files_status(output_dir: str, expected_relpaths: List[str]) -> Tuple[List[str], List[str]]:
    present = []
    missing = []

    root = Path(output_dir)

    for rel in expected_relpaths:
        p = root / rel
        if p.exists() and p.stat().st_size > 0:
            present.append(str(p))
        else:
            missing.append(str(p))

    return present, missing


def make_action(
    target: SystemTarget,
    analysis_type: str,
    target_name: str,
    action: str,
    reason: str,
    required_inputs: List[str],
    missing_inputs: List[str],
    expected_outputs: List[str],
    missing_outputs: List[str],
) -> AuditAction:
    return AuditAction(
        system_id=target.system_id,
        simulation_family=target.simulation_family,
        molecule_name=target.molecule_name,
        nmol=target.nmol,
        bilayer=target.bilayer,
        run_name=target.run_name,
        analysis_type=analysis_type,
        target=target_name,
        action=action,
        reason=reason,
        system_dir=target.system_dir,
        output_dir=target.output_dir,
        required_inputs="|".join(clean_paths(required_inputs)),
        missing_inputs="|".join(clean_paths(missing_inputs)),
        expected_outputs="|".join(clean_paths(expected_outputs)),
        missing_outputs="|".join(clean_paths(missing_outputs)),
    )


# =============================================================================
# AUDIT ACTION LOGIC
# =============================================================================

def audit_apl(target: SystemTarget) -> AuditAction:
    required_inputs = []

    if target.simulation_family == "gromacs_pull_relax":
        required_inputs = [target.pull_gro, target.pull_xtc, target.relax_gro, target.relax_xtc]
    elif target.simulation_family == "namd_cphmd":
        required_inputs = [target.namd_psf, target.namd_pdb, target.namd_dcd]

    required_inputs = clean_paths(required_inputs)
    missing_inputs = [p for p in required_inputs if not path_exists(p)]

    expected_outputs = [str(Path(target.output_dir) / rel) for rel in APL_OUTPUTS]
    _, missing_outputs = output_files_status(target.output_dir, APL_OUTPUTS)

    if not RUN_APL:
        return make_action(target, "APL", "bilayer", "skip", "RUN_APL is False",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if target.simulation_family != "gromacs_pull_relax":
        return make_action(target, "APL", "bilayer", "skip",
                           "APL execution currently implemented only for GROMACS Pull/Relax systems",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if not target.has_bilayer:
        return make_action(target, "APL", "bilayer", "skip",
                           "no bilayer detected; APL requires bilayer/membrane",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if not required_inputs:
        return make_action(target, "APL", "bilayer", "skip",
                           "no usable structure/trajectory inputs found for APL",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if missing_inputs:
        return make_action(target, "APL", "bilayer", "skip",
                           "missing required input files",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if FORCE_REANALYSE_APL:
        return make_action(target, "APL", "bilayer", "run",
                           "FORCE_REANALYSE_APL is True",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if missing_outputs:
        return make_action(target, "APL", "bilayer", "run",
                           "required APL outputs missing",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    input_mtime = newest_mtime(required_inputs)
    output_mtime = oldest_mtime(expected_outputs)

    if input_mtime > output_mtime:
        return make_action(target, "APL", "bilayer", "run",
                           "input files are newer than existing APL outputs",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    return make_action(target, "APL", "bilayer", "skip",
                       "APL outputs exist and appear current",
                       required_inputs, missing_inputs, expected_outputs, missing_outputs)


def sasa_target_is_possible(target: SystemTarget, sasa_target: str) -> Tuple[bool, str]:
    if sasa_target == "whole_system":
        return True, "whole_system allowed"

    if sasa_target == "bilayer":
        if target.has_bilayer:
            return True, "bilayer detected"
        return False, "SASA target bilayer selected but no bilayer detected"

    if sasa_target == "peptide":
        if target.has_peptide:
            return True, "peptide residues detected"
        return False, "SASA target peptide selected but no peptide residues detected"

    if sasa_target == "peptoid":
        if target.has_peptoid:
            return True, "peptoid residues detected"
        return False, "SASA target peptoid selected but no peptoid residues detected"

    if sasa_target == "polymer_all":
        if target.has_polymer:
            return True, "peptide and/or peptoid residues detected"
        return False, "SASA target polymer_all selected but no peptide/peptoid residues detected"

    return False, f"unknown SASA target: {sasa_target}"


def audit_sasa(target: SystemTarget, sasa_target: str) -> AuditAction:
    required_inputs = []

    if target.simulation_family == "gromacs_pull_relax":
        if target.has_relax:
            required_inputs.extend([target.relax_gro, target.relax_xtc])
        if target.has_pull:
            required_inputs.extend([target.pull_gro, target.pull_xtc])
    elif target.simulation_family == "namd_cphmd":
        required_inputs = [target.namd_dcd]
        if target.namd_psf:
            required_inputs.append(target.namd_psf)
        elif target.namd_pdb:
            required_inputs.append(target.namd_pdb)

    required_inputs = clean_paths(required_inputs)
    missing_inputs = [p for p in required_inputs if not path_exists(p)]

    output_relpaths = expected_sasa_outputs_for_target(sasa_target)
    expected_outputs = [str(Path(target.output_dir) / rel) for rel in output_relpaths]
    _, missing_outputs = output_files_status(target.output_dir, output_relpaths)

    if not RUN_SASA:
        return make_action(target, "SASA", sasa_target, "skip", "RUN_SASA is False",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if sasa_target not in VALID_SASA_TARGETS:
        return make_action(target, "SASA", sasa_target, "skip", "invalid SASA target in SASA_TARGETS",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    possible, possible_reason = sasa_target_is_possible(target, sasa_target)

    if not possible:
        return make_action(target, "SASA", sasa_target, "skip", possible_reason,
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if sasa_target != "bilayer" and not target.has_bilayer and not ALLOW_SASA_WITHOUT_BILAYER:
        return make_action(target, "SASA", sasa_target, "skip",
                           "system has no bilayer and ALLOW_SASA_WITHOUT_BILAYER is False",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if not required_inputs:
        return make_action(target, "SASA", sasa_target, "skip",
                           "no usable structure/trajectory inputs found for SASA",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if missing_inputs:
        return make_action(target, "SASA", sasa_target, "skip", "missing required input files",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if FORCE_REANALYSE_SASA:
        return make_action(target, "SASA", sasa_target, "run", "FORCE_REANALYSE_SASA is True",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if missing_outputs:
        return make_action(target, "SASA", sasa_target, "run", "required SASA outputs missing",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    input_mtime = newest_mtime(required_inputs)
    output_mtime = oldest_mtime(expected_outputs)

    if input_mtime > output_mtime:
        return make_action(target, "SASA", sasa_target, "run",
                           "input files are newer than existing SASA outputs",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    return make_action(target, "SASA", sasa_target, "skip",
                       "SASA outputs exist and appear current",
                       required_inputs, missing_inputs, expected_outputs, missing_outputs)


def audit_photos(target: SystemTarget) -> AuditAction:
    photo_mode = choose_photo_mode(target)

    if target.simulation_family == "gromacs_pull_relax":
        if target.has_relax:
            required_inputs = [target.relax_gro, target.relax_xtc]
        elif target.has_pull:
            required_inputs = [target.pull_gro, target.pull_xtc]
        else:
            required_inputs = []
    elif target.simulation_family == "namd_cphmd":
        required_inputs = []
        if target.namd_pdb:
            required_inputs.append(target.namd_pdb)
        if target.namd_dcd:
            required_inputs.append(target.namd_dcd)
    else:
        required_inputs = []

    if TCL_TEMPLATE.exists():
        required_inputs.append(str(TCL_TEMPLATE))

    required_inputs = clean_paths(required_inputs)
    missing_inputs = [p for p in required_inputs if not path_exists(p)]

    output_relpaths = PHOTO_OUTPUTS_BY_MODE.get(photo_mode, ["photos/photo_panel.png"])
    expected_outputs = [str(Path(target.output_dir) / rel) for rel in output_relpaths]
    _, missing_outputs = output_files_status(target.output_dir, output_relpaths)

    if not RUN_PHOTOS:
        return make_action(target, "PHOTO", photo_mode, "skip", "RUN_PHOTOS is False",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if photo_mode == "unknown":
        return make_action(target, "PHOTO", photo_mode, "skip",
                           "could not choose photo mode; no bilayer/polymer detected",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if not TCL_TEMPLATE.exists():
        return make_action(target, "PHOTO", photo_mode, "skip",
                           f"missing Tcl template: {TCL_TEMPLATE}",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    non_tcl_inputs = [p for p in required_inputs if not str(p).lower().endswith(".tcl")]

    if not non_tcl_inputs:
        return make_action(target, "PHOTO", photo_mode, "skip",
                           "no usable structure/trajectory inputs found for photo generation",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if missing_inputs:
        return make_action(target, "PHOTO", photo_mode, "skip", "missing required input files",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if FORCE_REGENERATE_PHOTOS:
        return make_action(target, "PHOTO", photo_mode, "run", "FORCE_REGENERATE_PHOTOS is True",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if missing_outputs:
        return make_action(target, "PHOTO", photo_mode, "run", "required photo outputs missing",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    input_mtime = newest_mtime(required_inputs)
    output_mtime = oldest_mtime(expected_outputs)

    if input_mtime > output_mtime:
        return make_action(target, "PHOTO", photo_mode, "run",
                           "input files are newer than existing photo outputs",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    return make_action(target, "PHOTO", photo_mode, "skip",
                       "photo outputs exist and appear current",
                       required_inputs, missing_inputs, expected_outputs, missing_outputs)


def audit_collages(target: SystemTarget) -> AuditAction:
    output_relpaths = expected_collage_outputs()
    expected_outputs = [str(Path(target.output_dir) / rel) for rel in output_relpaths]
    _, missing_outputs = output_files_status(target.output_dir, output_relpaths)

    required_inputs = []
    candidate_inputs = [
        Path(target.output_dir) / "photos/photo_panel.png",
        Path(target.output_dir) / "plots/apl_combined.png",
    ]

    for sasa_target in SASA_TARGETS:
        candidate_inputs.append(Path(target.output_dir) / f"plots/sasa_{sasa_target}_combined.png")

    for p in candidate_inputs:
        if p.exists() and p.stat().st_size > 0:
            required_inputs.append(str(p))

    required_inputs = clean_paths(required_inputs)
    missing_inputs: List[str] = []

    if not RUN_COLLAGES:
        return make_action(target, "COLLAGE", "mixed", "skip", "RUN_COLLAGES is False",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if not required_inputs:
        return make_action(target, "COLLAGE", "mixed", "skip",
                           "no source panels exist yet; run APL/SASA/photos first",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if FORCE_REGENERATE_COLLAGES:
        return make_action(target, "COLLAGE", "mixed", "run",
                           "FORCE_REGENERATE_COLLAGES is True and at least one source panel exists",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    if missing_outputs:
        return make_action(target, "COLLAGE", "mixed", "run", "one or more collage outputs missing",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    input_mtime = newest_mtime(required_inputs)
    output_mtime = oldest_mtime(expected_outputs)

    if input_mtime > output_mtime:
        return make_action(target, "COLLAGE", "mixed", "run",
                           "source panels are newer than existing collage outputs",
                           required_inputs, missing_inputs, expected_outputs, missing_outputs)

    return make_action(target, "COLLAGE", "mixed", "skip",
                       "collage outputs exist and appear current",
                       required_inputs, missing_inputs, expected_outputs, missing_outputs)


def build_audit_actions(targets: List[SystemTarget]) -> List[AuditAction]:
    actions: List[AuditAction] = []

    for target in targets:
        actions.append(audit_apl(target))

        for sasa_target in SASA_TARGETS:
            actions.append(audit_sasa(target, sasa_target))

        actions.append(audit_photos(target))
        actions.append(audit_collages(target))

    return actions


# =============================================================================
# APL EXECUTION
# =============================================================================

def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": PLOT_CFG["font_family"],
        "font.size": PLOT_CFG["font_size"],
        "axes.titlesize": PLOT_CFG["title_size"],
        "axes.labelsize": PLOT_CFG["axis_label_size"],
        "xtick.labelsize": PLOT_CFG["tick_label_size"],
        "ytick.labelsize": PLOT_CFG["tick_label_size"],
        "legend.fontsize": PLOT_CFG["legend_size"],
        "figure.dpi": PLOT_CFG["dpi"],
    })


def get_po4_and_leaflets(universe: mda.Universe, system_type: str) -> Dict[str, object]:
    po4 = universe.select_atoms(PO4_SELECTION)

    if len(po4) == 0:
        raise ValueError("No PO4 atoms found. Cannot calculate APL.")

    lookup_system_type = canonical_system_type(system_type)

    phospholipid_sel = "resname " + " ".join(sorted(PHOSPHOLIPID_RESNAMES))
    chol_sel = "resname " + " ".join(sorted(CHOL_RESNAMES))

    phospholipid_atoms = universe.select_atoms(phospholipid_sel)
    chol_atoms = universe.select_atoms(chol_sel)

    n_phospholipids = phospholipid_atoms.residues.n_residues
    n_chol = chol_atoms.residues.n_residues

    n_membrane_molecules = n_phospholipids + n_chol

    if "CHOL" in lookup_system_type and n_chol == 0:
        assumed_fraction = SYSTEM_TYPE_TO_ASSUMED_PHOSPHOLIPID_FRACTION.get(lookup_system_type)

        if assumed_fraction is None or assumed_fraction <= 0 or assumed_fraction > 1:
            raise ValueError(
                f"CHOL expected from system type but no CHOL residues detected and no fallback fraction available: "
                f"{system_type}"
            )

        n_membrane_molecules = n_phospholipids / assumed_fraction
        n_chol = n_membrane_molecules - n_phospholipids

    if n_membrane_molecules <= 0:
        raise ValueError("No membrane molecules counted.")

    lipids_per_leaflet = n_membrane_molecules / 2.0

    return {
        "po4": po4,
        "n_phospholipids": float(n_phospholipids),
        "n_chol": float(n_chol),
        "n_membrane_molecules": float(n_membrane_molecules),
        "lipids_per_leaflet": float(lipids_per_leaflet),
    }


def analyze_apl_trajectory(
    gro_path: Path,
    xtc_path: Path,
    phase_name: str,
    system_type: str,
) -> pd.DataFrame:
    u = mda.Universe(str(gro_path), str(xtc_path), in_memory=False)
    leaflet_info = get_po4_and_leaflets(u, system_type=system_type)

    po4 = leaflet_info["po4"]
    n_phospholipids = leaflet_info["n_phospholipids"]
    n_chol = leaflet_info["n_chol"]
    n_membrane_molecules = leaflet_info["n_membrane_molecules"]
    lipids_per_leaflet = leaflet_info["lipids_per_leaflet"]

    rows = []

    for ts in u.trajectory:
        dims = ts.dimensions

        if dims is None or len(dims) < 3:
            raise ValueError(f"{phase_name}: missing box dimensions.")

        lx, ly, lz = float(dims[0]), float(dims[1]), float(dims[2])

        if not all(np.isfinite([lx, ly, lz])) or min(lx, ly, lz) <= 0:
            raise ValueError(f"{phase_name}: invalid box dimensions: {dims}")

        coords = po4.positions.copy()

        coords[:, 0] = np.mod(coords[:, 0], lx)
        coords[:, 1] = np.mod(coords[:, 1], ly)
        coords[:, 2] = np.mod(coords[:, 2], lz)

        x_span = float(coords[:, 0].max() - coords[:, 0].min())
        y_span = float(coords[:, 1].max() - coords[:, 1].min())
        z_span = float(coords[:, 2].max() - coords[:, 2].min())

        time_ps = float(ts.time)
        time_ns = time_ps / 1000.0

        membrane_area = lx * ly

        apl_po4 = (x_span * y_span) / lipids_per_leaflet
        apl_box = membrane_area / lipids_per_leaflet

        rows.append({
            "phase": phase_name,
            "time_ps": time_ps,
            "time_ns": time_ns,

            "APL": apl_box,
            "APL_PO4": apl_po4,
            "APL_BOX": apl_box,

            "thickness_z_span": z_span,
            "lipids_per_leaflet": lipids_per_leaflet,
            "n_phospholipids": n_phospholipids,
            "n_chol": n_chol,
            "n_membrane_molecules": n_membrane_molecules,
            "membrane_area": membrane_area,
            "box_x": lx,
            "box_y": ly,
            "box_z": lz,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"{phase_name}: no trajectory frames found.")

    return df


def nearest_row(df: pd.DataFrame, target_ns: float) -> pd.Series:
    idx = (df["time_ns"] - target_ns).abs().idxmin()
    return df.loc[idx]


def last_10ns_stats(df: pd.DataFrame, end_time_ns: float) -> Tuple[float, float, float, float]:
    start_window = end_time_ns - 10.0
    window = df[df["time_ns"] >= start_window]

    if window.empty:
        return np.nan, np.nan, np.nan, np.nan

    return (
        float(window["APL_PO4"].mean()),
        float(window["APL_PO4"].std(ddof=0)),
        float(window["APL_BOX"].mean()),
        float(window["APL_BOX"].std(ddof=0)),
    )


def combine_pull_relax_timeseries(pull_df: pd.DataFrame, relax_df: pd.DataFrame) -> pd.DataFrame:
    pull_out = pull_df.copy()
    relax_out = relax_df.copy()

    pull_end = float(pull_out["time_ns"].max())

    pull_out["time_ns_combined"] = pull_out["time_ns"]
    pull_out["phase_combined"] = "Pull"

    relax_out["time_ns_combined"] = relax_out["time_ns"] + pull_end
    relax_out["phase_combined"] = "Relax"

    combined = pd.concat([pull_out, relax_out], ignore_index=True)
    combined = combined.sort_values("time_ns_combined").reset_index(drop=True)
    return combined


def plot_single_phase_apl(
    df: pd.DataFrame,
    output_png: Path,
    title: str,
    xlim,
    ylim,
    line_color: str,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_CFG["figure_size"], dpi=PLOT_CFG["dpi"])

    ax.plot(
        df["time_ns"],
        df["APL"],
        linewidth=PLOT_CFG["line_width"],
        color=line_color,
    )

    ax.set_title(wrap_title(title))
    ax.set_xlabel(PLOT_CFG["x_label"])
    ax.set_ylabel(PLOT_CFG["y_label"])

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if PLOT_CFG["tight_layout"]:
        fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def plot_combined_apl(
    pull_df: pd.DataFrame,
    relax_df: pd.DataFrame,
    output_png: Path,
) -> None:
    apply_plot_style()

    pull_plot = pull_df.copy()
    relax_plot = relax_df.copy()

    pull_end = float(pull_plot["time_ns"].max())
    pull_plot["time_ns_combined"] = pull_plot["time_ns"]
    relax_plot["time_ns_combined"] = relax_plot["time_ns"] + pull_end

    fig, ax = plt.subplots(figsize=PLOT_CFG["figure_size"], dpi=PLOT_CFG["dpi"])

    ax.plot(
        pull_plot["time_ns_combined"],
        pull_plot["APL"],
        linewidth=PLOT_CFG["line_width"],
        color=PLOT_CFG["pull_color"],
        label="Pull",
    )

    ax.plot(
        relax_plot["time_ns_combined"],
        relax_plot["APL"],
        linewidth=PLOT_CFG["line_width"],
        color=PLOT_CFG["relax_color"],
        label="Relax",
    )

    ax.set_title(wrap_title(PLOT_CFG["combined_title"]))
    ax.set_xlabel(PLOT_CFG["x_label"])
    ax.set_ylabel(PLOT_CFG["y_label"])
    ax.legend()

    if PLOT_CFG["combined_xlim"] is not None:
        ax.set_xlim(*PLOT_CFG["combined_xlim"])
    if PLOT_CFG["combined_ylim"] is not None:
        ax.set_ylim(*PLOT_CFG["combined_ylim"])

    if PLOT_CFG["tight_layout"]:
        fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def apply_combined_time_window(
    df: pd.DataFrame,
    start_ns: Optional[float],
    end_ns: Optional[float],
) -> pd.DataFrame:
    out = df.copy()

    if start_ns is not None:
        out = out[out["time_ns_combined"] >= start_ns]

    if end_ns is not None:
        out = out[out["time_ns_combined"] <= end_ns]

    return out.reset_index(drop=True)


def bin_apl_timeseries(
    df: pd.DataFrame,
    value_col: str,
    bin_size_ns: float,
    use_bin_center_for_x: bool,
    drop_incomplete_last_bin: bool,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Cannot bin an empty dataframe.")

    if bin_size_ns <= 0:
        raise ValueError("bin_size_ns must be > 0")

    if value_col not in df.columns:
        raise KeyError(f"Requested binning column not found: {value_col}")

    out = df.copy()

    cfg_start = BINNED_APL_CFG["start_ns"]
    cfg_end = BINNED_APL_CFG["end_ns"]

    tmin = float(out["time_ns_combined"].min()) if cfg_start is None else float(cfg_start)
    tmax = float(out["time_ns_combined"].max()) if cfg_end is None else float(cfg_end)

    out = out[(out["time_ns_combined"] >= tmin) & (out["time_ns_combined"] <= tmax)].copy()
    out.reset_index(drop=True, inplace=True)

    if out.empty:
        raise ValueError("No APL rows remain after applying binning time window.")

    out["bin_index"] = np.floor((out["time_ns_combined"] - tmin) / bin_size_ns).astype(int)
    out["bin_start_ns"] = tmin + out["bin_index"] * bin_size_ns
    out["bin_end_ns"] = out["bin_start_ns"] + bin_size_ns

    grouped = (
        out.groupby("bin_index", as_index=False)
        .agg(
            bin_start_ns=("bin_start_ns", "first"),
            bin_end_ns=("bin_end_ns", "first"),
            n_frames=(value_col, "size"),
            apl_mean=(value_col, "mean"),
            apl_std=(value_col, "std"),
            apl_min=(value_col, "min"),
            apl_max=(value_col, "max"),
            pull_frames=("phase_combined", lambda s: int((s == "Pull").sum())),
            relax_frames=("phase_combined", lambda s: int((s == "Relax").sum())),
        )
        .sort_values("bin_index")
        .reset_index(drop=True)
    )

    grouped["apl_std"] = grouped["apl_std"].fillna(0.0)

    grouped["phase_label"] = np.where(
        (grouped["pull_frames"] > 0) & (grouped["relax_frames"] == 0),
        "Pull",
        np.where(
            (grouped["relax_frames"] > 0) & (grouped["pull_frames"] == 0),
            "Relax",
            "Mixed",
        ),
    )

    if drop_incomplete_last_bin and len(grouped) > 0:
        last_row = grouped.iloc[-1]
        if float(last_row["bin_end_ns"]) > tmax + 1e-12:
            grouped = grouped.iloc[:-1].reset_index(drop=True)

    if use_bin_center_for_x:
        grouped["x_ns"] = 0.5 * (grouped["bin_start_ns"] + grouped["bin_end_ns"])
    else:
        grouped["x_ns"] = grouped["bin_start_ns"]

    return grouped


def compute_binned_slope(binned_df: pd.DataFrame) -> pd.DataFrame:
    if len(binned_df) < 2:
        return pd.DataFrame(columns=[
            "x_ns",
            "segment_start_ns",
            "segment_end_ns",
            "apl_start",
            "apl_end",
            "delta_apl",
            "delta_time_ns",
            "slope_apl_per_ns",
            "delta_apl_per_bin",
        ])

    rows = []

    for i in range(len(binned_df) - 1):
        a = binned_df.iloc[i]
        b = binned_df.iloc[i + 1]

        delta_time = float(b["x_ns"]) - float(a["x_ns"])
        delta_apl = float(b["apl_mean"]) - float(a["apl_mean"])
        slope = delta_apl / delta_time if delta_time != 0 else np.nan

        rows.append({
            "x_ns": 0.5 * (float(a["x_ns"]) + float(b["x_ns"])),
            "segment_start_ns": float(a["bin_start_ns"]),
            "segment_end_ns": float(b["bin_end_ns"]),
            "apl_start": float(a["apl_mean"]),
            "apl_end": float(b["apl_mean"]),
            "delta_apl": delta_apl,
            "delta_time_ns": delta_time,
            "slope_apl_per_ns": slope,
            "delta_apl_per_bin": delta_apl,
        })

    return pd.DataFrame(rows)


def get_slope_ylim(bin_size_ns: float) -> Tuple[float, float]:
    if BINNED_APL_CFG["slope_mode"] == "delta_per_bin":
        return tuple(BINNED_APL_CFG["slope_ylim_fixed"])

    apl_ymin, apl_ymax = BINNED_APL_CFG["apl_ylim"]
    max_abs = (apl_ymax - apl_ymin) / float(bin_size_ns)
    return (-max_abs, max_abs)


def plot_binned_apl(
    binned_df: pd.DataFrame,
    output_png: Path,
    system_label: str,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_CFG["figure_size"], dpi=PLOT_CFG["dpi"])

    x = binned_df["x_ns"].to_numpy()
    y = binned_df["apl_mean"].to_numpy()
    phase = binned_df["phase_label"].to_numpy()

    marker = "o" if BINNED_APL_CFG["show_apl_markers"] else None

    ax.plot(
        x,
        y,
        linewidth=PLOT_CFG["line_width"],
        color=BINNED_APL_CFG["apl_color"],
        marker=None,
        zorder=1,
    )

    pull_idx = np.where(phase == "Pull")[0]
    relax_idx = np.where(phase == "Relax")[0]

    if len(pull_idx) > 0:
        ax.plot(
            x[pull_idx],
            y[pull_idx],
            linewidth=PLOT_CFG["line_width"],
            color=BINNED_APL_CFG["pull_color"],
            marker=marker,
            markersize=BINNED_APL_CFG["apl_marker_size"],
            label="Pull",
            zorder=3,
        )

    if len(relax_idx) > 0:
        r0 = relax_idx[0]
        if r0 > 0:
            relax_plot_idx = np.concatenate(([r0 - 1], relax_idx))
        else:
            relax_plot_idx = relax_idx

        ax.plot(
            x[relax_plot_idx],
            y[relax_plot_idx],
            linewidth=PLOT_CFG["line_width"],
            color=BINNED_APL_CFG["relax_color"],
            marker=marker,
            markersize=BINNED_APL_CFG["apl_marker_size"],
            label="Relax",
            zorder=4,
        )

    if BINNED_APL_CFG["show_phase_boundary"]:
        ax.axvline(
            100.0,
            linestyle=BINNED_APL_CFG["phase_boundary_linestyle"],
            color=BINNED_APL_CFG["phase_boundary_color"],
            linewidth=BINNED_APL_CFG["phase_boundary_linewidth"],
            zorder=2,
        )

    if BINNED_APL_CFG["show_grid"]:
        ax.grid(True, alpha=0.3)

    ax.set_title(wrap_title(f"{system_label} | Average APL per {BINNED_APL_CFG['bin_size_ns']:g} ns bin"))
    ax.set_xlabel(BINNED_APL_CFG["apl_xlabel"])
    ax.set_ylabel(BINNED_APL_CFG["apl_ylabel"])

    if BINNED_APL_CFG["apl_xlim"] is not None:
        ax.set_xlim(*BINNED_APL_CFG["apl_xlim"])
    if BINNED_APL_CFG["apl_ylim"] is not None:
        ax.set_ylim(*BINNED_APL_CFG["apl_ylim"])

    if BINNED_APL_CFG["show_legend"]:
        ax.legend()

    if PLOT_CFG["tight_layout"]:
        fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def plot_binned_slope(
    slope_df: pd.DataFrame,
    output_png: Path,
    system_label: str,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_CFG["figure_size"], dpi=PLOT_CFG["dpi"])

    if BINNED_APL_CFG["slope_mode"] == "delta_per_bin":
        y_col = "delta_apl_per_bin"
        ylabel = f"ΔAPL per {BINNED_APL_CFG['bin_size_ns']:g} ns bin"
    else:
        y_col = "slope_apl_per_ns"
        ylabel = "Slope (APL per ns)"

    ax.set_title(wrap_title(f"{system_label} | {ylabel}"))
    ax.set_xlabel(BINNED_APL_CFG["slope_xlabel"])
    ax.set_ylabel(ylabel)

    if slope_df.empty:
        if BINNED_APL_CFG["slope_xlim"] is not None:
            ax.set_xlim(*BINNED_APL_CFG["slope_xlim"])

        ax.set_ylim(*get_slope_ylim(BINNED_APL_CFG["bin_size_ns"]))

        if BINNED_APL_CFG["show_grid"]:
            ax.grid(True, alpha=0.3)

        if PLOT_CFG["tight_layout"]:
            fig.tight_layout()

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png)
        plt.close(fig)
        return

    marker = "o" if BINNED_APL_CFG["show_slope_markers"] else None

    x = slope_df["x_ns"].to_numpy()
    y = slope_df[y_col].to_numpy()

    phase = np.where(x < 100.0, "Pull", "Relax")

    ax.plot(
        x,
        y,
        linewidth=PLOT_CFG["line_width"],
        color=BINNED_APL_CFG["apl_color"],
        marker=None,
        zorder=1,
    )

    pull_idx = np.where(phase == "Pull")[0]
    relax_idx = np.where(phase == "Relax")[0]

    if len(pull_idx) > 0:
        ax.plot(
            x[pull_idx],
            y[pull_idx],
            linewidth=PLOT_CFG["line_width"],
            color=BINNED_APL_CFG["pull_color"],
            marker=marker,
            markersize=BINNED_APL_CFG["slope_marker_size"],
            label="Pull",
            zorder=3,
        )

    if len(relax_idx) > 0:
        r0 = relax_idx[0]
        if r0 > 0:
            relax_plot_idx = np.concatenate(([r0 - 1], relax_idx))
        else:
            relax_plot_idx = relax_idx

        ax.plot(
            x[relax_plot_idx],
            y[relax_plot_idx],
            linewidth=PLOT_CFG["line_width"],
            color=BINNED_APL_CFG["relax_color"],
            marker=marker,
            markersize=BINNED_APL_CFG["slope_marker_size"],
            label="Relax",
            zorder=4,
        )

    if BINNED_APL_CFG["annotate_slope_values"]:
        for _, row in slope_df.iterrows():
            value = row[y_col]
            ax.annotate(
                BINNED_APL_CFG["annotation_fmt"].format(value),
                (row["x_ns"], value),
                fontsize=BINNED_APL_CFG["annotation_font_size"],
                rotation=BINNED_APL_CFG["annotation_rotation"],
                ha="center",
                va="bottom",
            )

    if BINNED_APL_CFG["show_phase_boundary"]:
        ax.axvline(
            100.0,
            linestyle=BINNED_APL_CFG["phase_boundary_linestyle"],
            color=BINNED_APL_CFG["phase_boundary_color"],
            linewidth=BINNED_APL_CFG["phase_boundary_linewidth"],
            zorder=2,
        )

    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)

    if BINNED_APL_CFG["show_grid"]:
        ax.grid(True, alpha=0.3)

    if BINNED_APL_CFG["slope_xlim"] is not None:
        ax.set_xlim(*BINNED_APL_CFG["slope_xlim"])

    ax.set_ylim(*get_slope_ylim(BINNED_APL_CFG["bin_size_ns"]))

    ymin, ymax = ax.get_ylim()
    major = BINNED_APL_CFG.get("slope_major_tick")
    minor = BINNED_APL_CFG.get("slope_minor_tick")

    if major is not None:
        ax.set_yticks(np.arange(ymin, ymax + major, major))
    if minor is not None:
        ax.set_yticks(np.arange(ymin, ymax + minor, minor), minor=True)
        ax.grid(True, which="minor", alpha=0.15)

    if BINNED_APL_CFG["show_legend"]:
        ax.legend()

    if PLOT_CFG["tight_layout"]:
        fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def save_binned_apl_outputs(
    combined_df: pd.DataFrame,
    data_dir: Path,
    plots_dir: Path,
    system_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric = BINNED_APL_CFG["metric"]

    windowed_df = apply_combined_time_window(
        combined_df,
        start_ns=BINNED_APL_CFG["start_ns"],
        end_ns=BINNED_APL_CFG["end_ns"],
    )

    binned_df = bin_apl_timeseries(
        df=windowed_df,
        value_col=metric,
        bin_size_ns=float(BINNED_APL_CFG["bin_size_ns"]),
        use_bin_center_for_x=bool(BINNED_APL_CFG["use_bin_center_for_x"]),
        drop_incomplete_last_bin=bool(BINNED_APL_CFG["drop_incomplete_last_bin"]),
    )

    slope_df = compute_binned_slope(binned_df)

    write_dataframe(binned_df, data_dir / "binned_apl.csv")
    write_dataframe(slope_df, data_dir / "binned_apl_slope.csv")

    if BINNED_APL_CFG["plot_enabled"]:
        plot_binned_apl(
            binned_df=binned_df,
            output_png=plots_dir / "binned_apl.png",
            system_label=system_label,
        )
        plot_binned_slope(
            slope_df=slope_df,
            output_png=plots_dir / "binned_apl_slope.png",
            system_label=system_label,
        )

    return binned_df, slope_df


def run_apl_for_target(target: SystemTarget) -> Dict[str, object]:
    if target.simulation_family != "gromacs_pull_relax":
        raise ValueError("APL execution is currently implemented only for GROMACS Pull/Relax systems.")

    system_dir = Path(target.system_dir)
    
    pull_gro = Path(target.pull_gro) if target.pull_gro else system_dir / "Pull.gro"
    pull_xtc = Path(target.pull_xtc) if target.pull_xtc else system_dir / "Pull.xtc"
    relax_gro = Path(target.relax_gro) if target.relax_gro else system_dir / "Relax.gro"
    relax_xtc = Path(target.relax_xtc) if target.relax_xtc else system_dir / "Relax.xtc"

    required = [pull_gro, pull_xtc, relax_gro, relax_xtc]
    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(f"Missing required APL input files: {missing}")

    output_root = Path(target.output_dir)
    data_dir = output_root / "data"
    plots_dir = output_root / "plots"

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"APL: reading Pull trajectory:  {pull_xtc}")
    pull_df = analyze_apl_trajectory(
        gro_path=pull_gro,
        xtc_path=pull_xtc,
        phase_name="pull",
        system_type=target.bilayer,
    )

    print(f"APL: reading Relax trajectory: {relax_xtc}")
    relax_df = analyze_apl_trajectory(
        gro_path=relax_gro,
        xtc_path=relax_xtc,
        phase_name="relax",
        system_type=target.bilayer,
    )

    nearest_100 = nearest_row(relax_df, TARGET_RELAX_TIME_NS)
    nearest_time_ns = float(nearest_100["time_ns"])

    if abs(nearest_time_ns - TARGET_RELAX_TIME_NS) > TARGET_RELAX_TOLERANCE_NS:
        raise ValueError(
            f"Relax trajectory does not reach {TARGET_RELAX_TIME_NS:g} ns within tolerance. "
            f"Nearest frame = {nearest_time_ns:.3f} ns; tolerance = {TARGET_RELAX_TOLERANCE_NS:g} ns"
        )

    start_row = pull_df.iloc[0]
    finish_row = nearest_100

    relax_end_ns = float(relax_df["time_ns"].max())

    (
        apl_po4_mean_last10,
        apl_po4_std_last10,
        apl_box_mean_last10,
        apl_box_std_last10,
    ) = last_10ns_stats(relax_df, relax_end_ns)

    combined_df = combine_pull_relax_timeseries(pull_df, relax_df)

    write_dataframe(combined_df, data_dir / "apl_timeseries.csv")

    plot_single_phase_apl(
        df=pull_df,
        output_png=plots_dir / "apl_pull.png",
        title=PLOT_CFG["pull_title"],
        xlim=PLOT_CFG["pull_xlim"],
        ylim=PLOT_CFG["pull_ylim"],
        line_color=PLOT_CFG["pull_color"],
    )

    plot_single_phase_apl(
        df=relax_df,
        output_png=plots_dir / "apl_relax.png",
        title=PLOT_CFG["relax_title"],
        xlim=PLOT_CFG["relax_xlim"],
        ylim=PLOT_CFG["relax_ylim"],
        line_color=PLOT_CFG["relax_color"],
    )

    plot_combined_apl(
        pull_df=pull_df,
        relax_df=relax_df,
        output_png=plots_dir / "apl_combined.png",
    )

    system_label = f"{target.molecule_name} | {pretty_bilayer_name(target.bilayer)} | {target.run_name}"

    binned_rows = np.nan
    slope_rows = np.nan

    if BINNED_APL_CFG["enabled"]:
        binned_df, slope_df = save_binned_apl_outputs(
            combined_df=combined_df,
            data_dir=data_dir,
            plots_dir=plots_dir,
            system_label=system_label,
        )
        binned_rows = len(binned_df)
        slope_rows = len(slope_df)

    start_apl_po4 = float(start_row["APL_PO4"])
    finish_apl_po4 = float(finish_row["APL_PO4"])
    start_apl_box = float(start_row["APL_BOX"])
    finish_apl_box = float(finish_row["APL_BOX"])

    summary = {
        "system_id": target.system_id,
        "simulation_family": target.simulation_family,
        "molecule_name": target.molecule_name,
        "nmol": target.nmol,
        "bilayer": target.bilayer,
        "run_name": target.run_name,
        "system_dir": target.system_dir,
        "output_dir": target.output_dir,

        "start_APL": start_apl_box,
        "finish_APL_100ns": finish_apl_box,
        "APL_change": finish_apl_box - start_apl_box,
        "APL_percent_change": ((finish_apl_box - start_apl_box) / start_apl_box * 100.0)
        if start_apl_box != 0 else np.nan,
        "APL_mean_last10ns": apl_box_mean_last10,
        "APL_std_last10ns": apl_box_std_last10,
        "APL_method": "BOX",

        "start_APL_PO4": start_apl_po4,
        "finish_APL_PO4_100ns": finish_apl_po4,
        "APL_change_PO4": finish_apl_po4 - start_apl_po4,
        "APL_percent_change_PO4": ((finish_apl_po4 - start_apl_po4) / start_apl_po4 * 100.0)
        if start_apl_po4 != 0 else np.nan,
        "APL_mean_last10ns_PO4": apl_po4_mean_last10,
        "APL_std_last10ns_PO4": apl_po4_std_last10,

        "start_APL_BOX": start_apl_box,
        "finish_APL_BOX_100ns": finish_apl_box,
        "APL_change_BOX": finish_apl_box - start_apl_box,
        "APL_percent_change_BOX": ((finish_apl_box - start_apl_box) / start_apl_box * 100.0)
        if start_apl_box != 0 else np.nan,
        "APL_mean_last10ns_BOX": apl_box_mean_last10,
        "APL_std_last10ns_BOX": apl_box_std_last10,

        "start_thickness": float(start_row["thickness_z_span"]),
        "finish_thickness_100ns": float(finish_row["thickness_z_span"]),
        "thickness_change": float(start_row["thickness_z_span"]) - float(finish_row["thickness_z_span"]),

        "lipids_per_leaflet": float(start_row["lipids_per_leaflet"]),
        "n_phospholipids": float(start_row["n_phospholipids"]),
        "n_chol": float(start_row["n_chol"]),
        "n_membrane_molecules": float(start_row["n_membrane_molecules"]),

        "membrane_area_start": float(start_row["membrane_area"]),
        "membrane_area_finish": float(finish_row["membrane_area"]),

        "relax_end_time_ns": relax_end_ns,
        "nearest_relax_time_to_target_ns": nearest_time_ns,
        "target_relax_time_ns": TARGET_RELAX_TIME_NS,
        "target_relax_tolerance_ns": TARGET_RELAX_TOLERANCE_NS,
        "reached_target_relax_time": True,

        "binned_apl_enabled": bool(BINNED_APL_CFG["enabled"]),
        "binned_rows": binned_rows,
        "slope_rows": slope_rows,

        "analysis_time": now_string(),
    }

    with open(output_root / "apl_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_apl_actions(
    actions: List[AuditAction],
    target_map: Dict[str, SystemTarget],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    apl_actions = [
        a for a in actions
        if a.analysis_type == "APL"
        and a.action == "run"
    ]

    summaries = []
    failures = []

    log_section("RUNNING APL ANALYSIS")
    log_kv("APL actions to run", len(apl_actions))

    for idx, action in enumerate(apl_actions, start=1):
        target = target_map.get(action.system_id)

        if target is None:
            failures.append({
                "system_id": action.system_id,
                "analysis_type": "APL",
                "reason": "target missing from target_map",
                "action_reason": action.reason,
                "system_dir": action.system_dir,
                "output_dir": action.output_dir,
                "failed_time": now_string(),
            })
            continue

        print(
            f"\n[{idx}/{len(apl_actions)}] "
            f"{target.molecule_name} | {target.bilayer} | {target.run_name}"
        )

        try:
            summary = run_apl_for_target(target)
            summaries.append(summary)
            print("  -> APL complete")

        except Exception as exc:
            failures.append({
                "system_id": target.system_id,
                "simulation_family": target.simulation_family,
                "molecule_name": target.molecule_name,
                "nmol": target.nmol,
                "bilayer": target.bilayer,
                "run_name": target.run_name,
                "analysis_type": "APL",
                "reason": f"{type(exc).__name__}: {exc}",
                "action_reason": action.reason,
                "system_dir": target.system_dir,
                "output_dir": target.output_dir,
                "failed_time": now_string(),
            })
            print(f"  -> APL failed: {type(exc).__name__}: {exc}")

    summary_df = pd.DataFrame(summaries)
    failed_df = pd.DataFrame(failures)

    if not summary_df.empty:
        append_dataframe_to_csv_dedup(
            summary_df,
            APL_SUMMARY_CSV,
            dedup_subset=["system_id"],
        )

    if not failed_df.empty:
        append_dataframe_to_csv_dedup(
            failed_df,
            APL_FAILED_CSV,
            dedup_subset=["system_id", "analysis_type", "reason"],
        )

    return summary_df, failed_df


# =============================================================================
# REPORT WRITING
# =============================================================================

def target_to_input_rows(target: SystemTarget) -> List[Dict[str, object]]:
    rows = []

    file_map = {
        "pull_gro": target.pull_gro,
        "pull_xtc": target.pull_xtc,
        "relax_gro": target.relax_gro,
        "relax_xtc": target.relax_xtc,
        "pull_cpt": target.pull_cpt,
        "relax_cpt": target.relax_cpt,
        "namd_psf": target.namd_psf,
        "namd_pdb": target.namd_pdb,
        "namd_dcd": target.namd_dcd,
        "namd_xst": target.namd_xst,
        "namd_log": target.namd_log,
        "representative_structure": target.representative_structure,
        "representative_trajectory": target.representative_trajectory,
    }

    for label, p in file_map.items():
        rows.append({
            "system_id": target.system_id,
            "simulation_family": target.simulation_family,
            "molecule_name": target.molecule_name,
            "nmol": target.nmol,
            "bilayer": target.bilayer,
            "run_name": target.run_name,
            "file_role": label,
            "path": p,
            "exists": path_exists(p),
            "size_bytes": file_size(p),
        })

    return rows


def write_audit_reports(targets: List[SystemTarget], actions: List[AuditAction]) -> Dict[str, object]:
    target_rows = [asdict(t) for t in targets]
    actions_rows = [asdict(a) for a in actions]

    input_rows: List[Dict[str, object]] = []
    for target in targets:
        input_rows.extend(target_to_input_rows(target))

    systems_df = pd.DataFrame(target_rows)
    actions_df = pd.DataFrame(actions_rows)
    inputs_df = pd.DataFrame(input_rows)

    existing_output_rows = []

    for action in actions:
        expected = [x for x in action.expected_outputs.split("|") if x]
        missing = set(x for x in action.missing_outputs.split("|") if x)

        for p in expected:
            existing_output_rows.append({
                "system_id": action.system_id,
                "analysis_type": action.analysis_type,
                "target": action.target,
                "path": p,
                "exists": p not in missing,
                "size_bytes": file_size(p),
            })

    outputs_df = pd.DataFrame(existing_output_rows)

    write_dataframe(systems_df, AUDIT_SYSTEMS_CSV)
    write_dataframe(inputs_df, AUDIT_INPUTS_CSV)
    write_dataframe(outputs_df, AUDIT_OUTPUTS_CSV)
    write_dataframe(actions_df, AUDIT_ACTIONS_CSV)

    summary = {
        "script_version": SCRIPT_VERSION,
        "audit_time": now_string(),
        "analysis_dir": str(ANALYSIS_DIR),
        "data_root": str(DATA_ROOT),
        "results_root": str(RESULTS_ROOT),
        "target_dir_mode": TARGET_DIR_MODE,
        "target_dirs": TARGET_DIRS,
        "cphmd_roots": CPHMD_ROOTS,
        "n_systems": len(targets),
        "n_actions": len(actions),
        "n_actions_run": int(sum(1 for a in actions if a.action == "run")),
        "n_actions_skip": int(sum(1 for a in actions if a.action == "skip")),
        "settings": {
            "RUN_AUDIT": RUN_AUDIT,
            "FORCE_FULL_AUDIT": FORCE_FULL_AUDIT,
            "TRUST_PREVIOUS_STATE_JSON": TRUST_PREVIOUS_STATE_JSON,
            "RUN_ANALYSIS_AFTER_AUDIT": RUN_ANALYSIS_AFTER_AUDIT,
            "RUN_APL": RUN_APL,
            "RUN_SASA": RUN_SASA,
            "RUN_PHOTOS": RUN_PHOTOS,
            "RUN_COLLAGES": RUN_COLLAGES,
            "RUN_GROUPED_STACKS": RUN_GROUPED_STACKS,
            "RUN_GROMACS_PULL_RELAX": RUN_GROMACS_PULL_RELAX,
            "RUN_NAMD_CPHMD": RUN_NAMD_CPHMD,
            "SASA_TARGETS": SASA_TARGETS,
            "SASA_FRAME_MODE": SASA_FRAME_MODE,
            "PHOTO_MODE": PHOTO_MODE,
            "EXCLUDED_DIR_NAMES": sorted(EXCLUDED_DIR_NAMES),
            "EXCLUDED_DIR_NAME_CONTAINS": sorted(EXCLUDED_DIR_NAME_CONTAINS),
            "TARGET_RELAX_TIME_NS": TARGET_RELAX_TIME_NS,
            "TARGET_RELAX_TOLERANCE_NS": TARGET_RELAX_TOLERANCE_NS,
            "BINNED_APL_CFG": BINNED_APL_CFG,
            "PLOT_CFG": PLOT_CFG,
        },
        "report_paths": {
            "audit_systems_csv": str(AUDIT_SYSTEMS_CSV),
            "audit_inputs_csv": str(AUDIT_INPUTS_CSV),
            "audit_outputs_csv": str(AUDIT_OUTPUTS_CSV),
            "audit_actions_csv": str(AUDIT_ACTIONS_CSV),
            "audit_summary_json": str(AUDIT_SUMMARY_JSON),
            "analysis_state_json": str(ANALYSIS_STATE_JSON),
            "apl_summary_csv": str(APL_SUMMARY_CSV),
            "apl_failed_csv": str(APL_FAILED_CSV),
        },
    }

    with open(AUDIT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    state_payload = {
        "last_updated": now_string(),
        "script_version": SCRIPT_VERSION,
        "targets": target_rows,
        "actions": actions_rows,
        "summary": summary,
    }

    with open(ANALYSIS_STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state_payload, f, indent=2)

    return summary


# =============================================================================
# VALIDATION / DISCOVERY CONTROLLER
# =============================================================================

def validate_settings() -> None:
    bad_sasa = [x for x in SASA_TARGETS if x not in VALID_SASA_TARGETS]
    if bad_sasa:
        raise ValueError(f"Invalid SASA_TARGETS: {bad_sasa}. Allowed: {sorted(VALID_SASA_TARGETS)}")

    if SASA_FRAME_MODE not in {"start_end", "stride", "all"}:
        raise ValueError("SASA_FRAME_MODE must be one of: start_end, stride, all")

    if PHOTO_MODE not in {"auto", "bilayer", "waterbox", "polymer_only"}:
        raise ValueError("PHOTO_MODE must be one of: auto, bilayer, waterbox, polymer_only")

    if TARGET_DIR_MODE and not TARGET_DIRS:
        raise ValueError("TARGET_DIR_MODE=True but TARGET_DIRS is empty.")

    if RUN_NAMD_CPHMD and not CPHMD_ROOTS:
        print("NOTE: RUN_NAMD_CPHMD=True but CPHMD_ROOTS is empty. CpHMD scan will find nothing.")


def discover_all_targets() -> List[SystemTarget]:
    targets: List[SystemTarget] = []

    if RUN_GROMACS_PULL_RELAX:
        log_section("DISCOVERING GROMACS PULL/RELAX SYSTEMS")
        gromacs_targets = discover_gromacs_targets()
        log_kv("GROMACS targets found", len(gromacs_targets))
        targets.extend(gromacs_targets)

    if RUN_NAMD_CPHMD and CPHMD_ROOTS:
        log_section("DISCOVERING NAMD CPHMD SYSTEMS")
        cphmd_targets = discover_namd_cphmd_targets()
        log_kv("CpHMD targets found", len(cphmd_targets))
        targets.extend(cphmd_targets)

    unique: Dict[str, SystemTarget] = {}
    for t in targets:
        unique[t.system_id] = t

    return list(unique.values())


def print_summary(summary: Dict[str, object], actions: List[AuditAction]) -> None:
    log_section("AUDIT SUMMARY")
    log_kv("Systems found", summary["n_systems"])
    log_kv("Audit actions", summary["n_actions"])
    log_kv("Actions marked RUN", summary["n_actions_run"])
    log_kv("Actions marked SKIP", summary["n_actions_skip"])

    if actions:
        df = pd.DataFrame([asdict(a) for a in actions])
        grouped = (
            df.groupby(["analysis_type", "action"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["analysis_type", "action"])
        )

        print("\nAction breakdown:")
        print(grouped.to_string(index=False))

    print("\nReports written:")
    log_kv("Systems", AUDIT_SYSTEMS_CSV)
    log_kv("Inputs", AUDIT_INPUTS_CSV)
    log_kv("Existing outputs", AUDIT_OUTPUTS_CSV)
    log_kv("Actions needed", AUDIT_ACTIONS_CSV)
    log_kv("Audit JSON", AUDIT_SUMMARY_JSON)
    log_kv("State JSON", ANALYSIS_STATE_JSON)


# =============================================================================
# EXECUTION CONTROLLER
# =============================================================================

def run_analysis_from_audit(
    actions: List[AuditAction],
    targets: List[SystemTarget],
) -> None:
    target_map = {t.system_id: t for t in targets}

    if RUN_APL:
        summary_df, failed_df = run_apl_actions(actions, target_map)
        log_section("APL EXECUTION SUMMARY")
        log_kv("APL completed", len(summary_df))
        log_kv("APL failed", len(failed_df))
        log_kv("APL summary CSV", APL_SUMMARY_CSV)
        log_kv("APL failed CSV", APL_FAILED_CSV)

    if RUN_SASA:
        print("NOTE: SASA execution is not implemented in v0.3. It remains audit-only.")

    if RUN_PHOTOS:
        print("NOTE: Photo execution is not implemented in v0.3. It remains audit-only.")

    if RUN_COLLAGES:
        print("NOTE: Collage execution is not implemented in v0.3. It remains audit-only.")


# =============================================================================
# COMMAND LINE SUPPORT
# =============================================================================

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PFPeptoids analysis suite v0.3.")

    parser.add_argument("--audit", action="store_true", help="Run audit/discovery mode.")
    parser.add_argument("--run-from-audit", action="store_true", help="Run enabled analyses after audit.")
    parser.add_argument("--no-confirm", action="store_true", help="Do not ask for confirmation after audit.")
    parser.add_argument("--target-dir", action="append", default=None, help="Add/override target directory.")
    parser.add_argument("--cphmd-root", action="append", default=None, help="Add CpHMD root directory.")

    args, unknown = parser.parse_known_args(argv)

    if unknown:
        print(f"NOTE: Ignoring unknown command-line arguments: {unknown}")

    return args


def apply_cli_overrides(args: argparse.Namespace) -> None:
    global RUN_AUDIT
    global RUN_ANALYSIS_AFTER_AUDIT
    global REQUIRE_USER_CONFIRMATION_AFTER_AUDIT
    global TARGET_DIRS
    global TARGET_DIR_MODE
    global CPHMD_ROOTS

    if args.audit:
        RUN_AUDIT = True

    if args.run_from_audit:
        RUN_ANALYSIS_AFTER_AUDIT = True

    if args.no_confirm:
        REQUIRE_USER_CONFIRMATION_AFTER_AUDIT = False

    if args.target_dir:
        TARGET_DIR_MODE = True
        TARGET_DIRS = args.target_dir

    if args.cphmd_root:
        CPHMD_ROOTS = args.cphmd_root


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    apply_cli_overrides(args)

    mkdirs()
    validate_settings()

    log_section("PFPEPTOIDS ANALYSIS SUITE")
    log_kv("Version", SCRIPT_VERSION)
    log_kv("Started", now_string())
    log_kv("Analysis dir", ANALYSIS_DIR)
    log_kv("Data root", DATA_ROOT)
    log_kv("Results root", RESULTS_ROOT)
    log_kv("Target dir mode", TARGET_DIR_MODE)
    log_kv("Target dirs", TARGET_DIRS)
    log_kv("CpHMD roots", CPHMD_ROOTS)
    log_kv("Excluded dir names", sorted(EXCLUDED_DIR_NAMES))
    log_kv("Excluded path contains", sorted(EXCLUDED_DIR_NAME_CONTAINS))
    log_kv("RUN_APL", RUN_APL)
    log_kv("RUN_SASA", RUN_SASA)
    log_kv("SASA_TARGETS", SASA_TARGETS)
    log_kv("RUN_PHOTOS", RUN_PHOTOS)
    log_kv("PHOTO_MODE", PHOTO_MODE)
    log_kv("RUN_COLLAGES", RUN_COLLAGES)
    log_kv("RUN_ANALYSIS_AFTER_AUDIT", RUN_ANALYSIS_AFTER_AUDIT)

    if not RUN_AUDIT:
        print("RUN_AUDIT is False. Nothing to do.")
        return

    targets = discover_all_targets()
    actions = build_audit_actions(targets)
    summary = write_audit_reports(targets, actions)

    print_summary(summary, actions)

    if not RUN_ANALYSIS_AFTER_AUDIT:
        log_section("STOPPING AFTER AUDIT")
        print("Set RUN_ANALYSIS_AFTER_AUDIT = True to execute enabled analysis modules.")
        print("For v0.3, only APL execution is implemented.")
        return

    if REQUIRE_USER_CONFIRMATION_AFTER_AUDIT:
        print("\nThe audit has completed.")
        print("No execution has started yet.")
        response = input("Type YES to proceed to enabled analysis execution: ").strip()
        if response != "YES":
            print("Analysis execution cancelled.")
            return

    run_analysis_from_audit(actions, targets)

    log_section("RUN COMPLETE")
    print("Re-run the script with RUN_ANALYSIS_AFTER_AUDIT = False to refresh the audit after execution.")


if __name__ == "__main__":
    main()