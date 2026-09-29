#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APL analysis for PFPeptoids systems.

Creates
-------
Core outputs:
1) masterdataanalysis.csv
2) bilayercomparisonanalysis.csv
3) failed_systems.csv
4) analysis_metadata.json
5) Per-system CSV + PNG outputs grouped by molecule/bilayer/run

Per-system time-series outputs:
6) pull_APL_vs_time.png
7) relax_APL_vs_time.png
8) combined_APL_over_time.png
9) APL_time_series.csv / .xlsx

Per-system binned outputs:
10) binned_APL.csv / .xlsx
11) binned_APL_slope.csv / .xlsx
12) binned_APL.png
13) binned_APL_slope.png
14) overlay_binned_APL_all_bilayers_run_<run>_nmol_<nmol>.png
15) overlay_binned_APL_2bilayers_<pair>_run_<run>_nmol_<nmol>.png

Author-facing notes
-------------------
- This script never overwrites completed master rows already present in masterdataanalysis.csv
- It analyzes only new valid systems for master-row insertion
- It can still regenerate missing plots/derived outputs for previously analyzed systems
- It stores BOTH APL methods:
    a) PO4-spread APL
    b) box-area APL
- Start APL comes from first frame of pull.xtc
- Finish APL comes from the frame nearest 100 ns in relax.xtc
- If relax does not reach 100 ns within tolerance, the system is written to failed_systems.csv
- Molecular weights are dictionary-based for now
- Peptoid monomer masses can be provided in JSON without changing this script

Script location expected:
    /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/aplanalysis.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

import MDAnalysis as mda
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D
import seaborn as sns

import math
import textwrap



# =============================================================================
# USER CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# TARGET DIRECTORY / MANUAL INPUT CONTROL
# -----------------------------------------------------------------------------
# PFPEPTOIDS_APL_RUN_SELECTOR_V1
#
# Normal active-learning use:
#     TARGET_SELECTION_MODE = "run"
#     ANALYSIS_RUN = "ML_cycle_01"
#
# This analyzes ONLY:
#     <Boxes/2.1>/<bilayer>/run_ML_cycle_01/
# for every bilayer listed in ANALYSIS_BILAYERS.
#
# The cumulative masterdataanalysis.csv remains unchanged as the single master
# table. Each row already stores run_number + system_dir, so run_0,
# run_ML_cycle_01, run_ML_cycle_02, etc. remain distinct provenance records.
#
# Supported modes:
#   "run"      = SAFE DEFAULT. Analyze one named run across selected bilayers.
#   "custom"   = Analyze only paths explicitly listed in CUSTOM_TARGET_DIRS.
#   "all_runs" = Recursively scan selected bilayer roots. This is intentionally
#                blocked unless ALLOW_ALL_RUNS_SCAN is also set True.
#
# MANUAL_MODE below is still available for rare one-system/direct-file work.
# Do not enable MANUAL_MODE and TARGET_DIR_MODE at the same time.
# -----------------------------------------------------------------------------
TARGET_DIR_MODE = True

TARGET_SELECTION_MODE = "run"  # "run", "custom", or "all_runs"

# For normal ML-cycle analysis change ONLY this value between cycles.
# Both "ML_cycle_01" and "run_ML_cycle_01" are accepted; internally the
# leading "run_" is removed before paths are constructed.
ANALYSIS_RUN = "ML_cycle_01"

# Bilayers that form the DDAPL pair used by the active-learning workflow.
# Additional bilayers can be added deliberately if required later.
ANALYSIS_BILAYERS = [
    "POPC60CHOL40_W_WF",
    "POPE75POPG25_W_WF",
]

TARGET_DATA_ROOT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1"
)

# Used only when TARGET_SELECTION_MODE = "custom".
# These may be final system folders, run folders, bilayer folders, or other
# explicit roots already supported by scan_target_directories().
CUSTOM_TARGET_DIRS = [
    # "/full/path/to/a/specific/run_or_system",
]

# Deliberate protection against accidentally discovering old valid run_* data.
# To scan every run under ANALYSIS_BILAYERS you must set BOTH:
#     TARGET_SELECTION_MODE = "all_runs"
#     ALLOW_ALL_RUNS_SCAN = True
ALLOW_ALL_RUNS_SCAN = False


def _normalise_analysis_run_label(value: str) -> str:
    label = str(value).strip()
    if label.startswith("run_"):
        label = label[4:]
    if not label:
        raise ValueError("ANALYSIS_RUN cannot be empty.")
    if "/" in label or "\\" in label:
        raise ValueError(
            "ANALYSIS_RUN must be a run label only, not a path. "
            "Example: 'ML_cycle_01'."
        )
    return label


NORMALIZED_ANALYSIS_RUN = _normalise_analysis_run_label(ANALYSIS_RUN)

if TARGET_SELECTION_MODE == "run":
    _selected_run_folder = f"run_{NORMALIZED_ANALYSIS_RUN}"
    TARGET_DIRS = [
        str(TARGET_DATA_ROOT / bilayer / _selected_run_folder)
        for bilayer in ANALYSIS_BILAYERS
    ]

elif TARGET_SELECTION_MODE == "custom":
    if not CUSTOM_TARGET_DIRS:
        raise ValueError(
            "TARGET_SELECTION_MODE='custom' but CUSTOM_TARGET_DIRS is empty."
        )
    TARGET_DIRS = [str(Path(path)) for path in CUSTOM_TARGET_DIRS]

elif TARGET_SELECTION_MODE == "all_runs":
    if not ALLOW_ALL_RUNS_SCAN:
        raise ValueError(
            "TARGET_SELECTION_MODE='all_runs' is intentionally locked. "
            "Set ALLOW_ALL_RUNS_SCAN=True only when you explicitly want to "
            "scan historical run_* folders as well."
        )
    TARGET_DIRS = [
        str(TARGET_DATA_ROOT / bilayer)
        for bilayer in ANALYSIS_BILAYERS
    ]

else:
    raise ValueError(
        "Unknown TARGET_SELECTION_MODE={!r}. Expected 'run', 'custom', or "
        "'all_runs'.".format(TARGET_SELECTION_MODE)
    )

MANUAL_MODE = False
MANUAL_SYSTEM_DIR = None

MANUAL_PULL_GRO = None
MANUAL_PULL_XTC = None
MANUAL_RELAX_GRO = None
MANUAL_RELAX_XTC = None

MANUAL_MOLECULE_NAME = "manual_system_1"
MANUAL_SYSTEM_TYPE = "manual_bilayer"
MANUAL_RUN_NUMBER = "manual"

SCRIPT_PATH = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/aplanalysis.py")
ANALYSIS_DIR = SCRIPT_PATH.parent
DATA_ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1")

MASTER_CSV = ANALYSIS_DIR / "masterdataanalysis.csv"
COMPARISON_CSV = ANALYSIS_DIR / "bilayercomparisonanalysis.csv"
FAILED_CSV = ANALYSIS_DIR / "failed_systems.csv"
METADATA_JSON = ANALYSIS_DIR / "analysis_metadata.json"
OUTPUT_ROOT = ANALYSIS_DIR / "output_2"

# If present, this JSON will be used to extend/override the built-in mass dictionaries.
# Example format:
# {
#   "peptide": {"X": 123.45},
#   "peptoid": {"Na": 101.23, "Nk": 144.21}
# }
MASS_OVERRIDE_JSON = ANALYSIS_DIR / "monomer_masses.json"

# Relax trajectory must contain a frame sufficiently close to 100 ns
TARGET_RELAX_TIME_NS = 100.0
TARGET_RELAX_TOLERANCE_NS = 0.5

# Use PO4 beads for PO4-style APL and thickness nd to calculate the CHOL fraction
PO4_SELECTION = "name PO4"
PHOSPHOLIPID_RESNAMES = {"POPC", "POPS", "POPE"}
CHOL_RESNAMES = {"CHOL"}
SYSTEM_TYPE_TO_ASSUMED_PHOSPHOLIPID_FRACTION = {
    "POPC60CHOL40_W_WF": 0.60,
    "POPC80POPS20_W_WF": 1.00,
    "POPE75POPG25_W_WF": 1.00,
}

SYSTEM_TYPE_ALIASES = {
    "POPC60CHOL40_20Az": "POPC60CHOL40_W_WF",
    "POPE75POPG25_20Az": "POPE75POPG25_W_WF",
    "POPC80POPS20_20Az": "POPC80POPS20_W_WF",
}


def canonical_system_type(system_type: str) -> str:
    return SYSTEM_TYPE_ALIASES.get(system_type, system_type)

# Per-molecule bilayer comparison order
BILAYER_PRIORITY = [
    "POPC60CHOL40_20Az",
    "POPE75POPG25_20Az",
    "POPC80POPS20_20Az",
    "POPC60CHOL40_W_WF",  # POPC:CHOL
    "POPE75POPG25_W_WF",  # POPE:POPG
    "POPC80POPS20_W_WF",  # POPC:POPS
]

# Extra two-bilayer stacked images generated in addition to the normal all-bilayer images.
# Default = the two bilayers you had active in the reduced TARGET_DIRS block.
# Add more tuples here if you want more two-bilayer versions.
TWO_BILAYER_STACKS = [
    ("POPC60CHOL40_W_WF", "POPE75POPG25_W_WF"),
    # ("POPC60CHOL40_W_WF", "POPC80POPS20_W_WF"),
    # ("POPE75POPG25_W_WF", "POPC80POPS20_W_WF"),
]

GENERATE_TWO_BILAYER_STACKS = True

# Extra overlaid binned-APL graphs generated in addition to the solo images
# and the stacked images.
#
# These put multiple bilayers onto ONE axis so POPC:CHOL and POPE:POPG
# can be compared directly.
GENERATE_OVERLAY_BILAYER_GRAPHS = True

# If True, create one overlaid graph containing all bilayers present for the
# same molecule/run/nmol.
GENERATE_ALL_BILAYER_OVERLAY = False

# If True, create selected two-bilayer overlaid graphs using TWO_BILAYER_STACKS.
GENERATE_TWO_BILAYER_OVERLAYS = True

# -----------------------------------------------------------------------------
# Overlay plot output / appearance controls
# -----------------------------------------------------------------------------

# File naming for overlay plots.
# You can rename these without changing the rest of the script.
OVERLAY_ALL_BILAYERS_FILENAME_TEMPLATE = (
    "{molecule_name}_overlay_all_bilayers_run_{run_number}_nmol_{nmol}.png"
)

OVERLAY_TWO_BILAYERS_FILENAME_TEMPLATE = (
    "{molecule_name}_overlay_{pair_label}_run_{run_number}_nmol_{nmol}.png"
)

# Set False to remove the title from overlay plots.
OVERLAY_SHOW_TITLE = False

# Figure size for overlay plots only.
# Increase the first number to make more horizontal room for the legend.
OVERLAY_FIGURE_SIZE = (10.5, 5.5)

# Legend controls for overlay plots only.
# Forced inside the graph area at the top-right corner.
OVERLAY_LEGEND_LOC = "upper right"
OVERLAY_LEGEND_BBOX_TO_ANCHOR = (0.98, 0.98)
OVERLAY_LEGEND_FONT_SIZE = 18
OVERLAY_LEGEND_FRAME_ALPHA = 0.9

# When the legend is outside the axes, this prevents it being cut off.
OVERLAY_SAVE_BBOX_INCHES = "tight"

# Overlay graph styling.
# IMPORTANT:
#   - colour = phase
#   - marker/symbol = bilayer
#   - "Mixed" is intentionally NOT plotted
OVERLAY_PHASE_COLORS = {
    "Pull": "#554768",
    "Relax": "#81B0CD",
}

OVERLAY_BILAYER_MARKERS = {
    "POPC60CHOL40_W_WF": "o",   # POPC:CHOL
    "POPE75POPG25_W_WF": "s",   # POPE:POPG
    "POPC80POPS20_W_WF": "^",   # POPC:POPS
}

# Kept for metadata / backwards compatibility.
OVERLAY_BILAYER_COLORS = {}
OVERLAY_PHASE_LINESTYLES = {
    "Pull": "-",
    "Relax": "-",
}
OVERLAY_PHASE_MARKERS = {}

# Tick spacing for overlay plots.
OVERLAY_X_TICK_STEP = 50.0
OVERLAY_Y_TICK_STEP = 10.0

# Slightly smaller title for the overlay graph to keep it clean if titles are enabled.
OVERLAY_TITLE_FONT_SIZE = 20
OVERLAY_TITLE_Y = 1.03
OVERLAY_TITLE_PAD = 14

# -----------------------------------------------------------------------------
# Plot styling — all user-customizable
# -----------------------------------------------------------------------------
font_manager.fontManager.addfont("/users/pfb19164/Documents/Fonts/centurygothic.ttf")
plt.rcParams['font.family'] = 'Century Gothic'
sns.set_context("poster")

PLOT_CFG = {
    "style_use_default_matplotlib": True,
    "font_family": "Century Gothic",
    "font_size": 20,
    "title_size": 24,
    "axis_label_size": 24,
    "tick_label_size": 20,
    "legend_size": 24,

    # Set False to remove titles from standard plots.
    "show_titles": False,

    # Title wrapping / spacing controls.
    # Lower width = title wraps sooner.
    "title_wrap_width": 30,
    "subplot_title_wrap_width": 28,
    "suptitle_wrap_width": 42,
    "title_pad": 18,
    "suptitle_y": 0.98,

    "line_width": 2.0,
    "figure_size": (8.0, 5.0),
    "dpi": 300,
    "pull_title": "Pull APL vs Time",
    "relax_title": "Relax APL vs Time",
    "combined_title": "Combined APL vs Time",
    "stacked_combined_title": "Combined APL Comparison",

    "x_label": "Time (ns)",
    "y_label_po4": "APL (Å²)",
    "y_label_box": "APL (Å²)",

    "combined_use_box_method": True,   # False = PO4, True = BOX
    "single_use_box_method": True,     # False = PO4, True = BOX

    "pull_color": "#554768",
    "relax_color": "#81B0CD",

    "pull_xlim": None,
    "relax_xlim": None,
    "combined_xlim": None,

    "pull_ylim": (40, 90),
    "relax_ylim": (40, 90),
    "combined_ylim": (40, 90),

    "tight_layout": True,
}


# =============================================================================
# PLOT_CFG = {
#     "style_use_default_matplotlib": True,
#     "font_family": "DejaVu Sans",
#     "font_size": 12,
#     "title_size": 14,
#     "axis_label_size": 13,
#     "tick_label_size": 11,
#     "legend_size": 10,
#     "line_width": 2.0,
#     "figure_size": (8.0, 5.0),
#     "dpi": 300,
# 
#     "pull_title": "Pull APL vs Time",
#     "relax_title": "Relax APL vs Time",
#     "combined_title": "Combined APL vs Time",
#     "stacked_combined_title": "Combined APL Comparison",
# 
#     "x_label": "Time (ns)",
#     "y_label_po4": "APL (Å²)",
#     "y_label_box": "APL (Å²)",
# 
#     "combined_use_box_method": True,   # False = PO4, True = BOX
#     "single_use_box_method": True,     # False = PO4, True = BOX
# 
#     "pull_color": "#554768",
#     "relax_color": "#81B0CD",
# 
#     "pull_xlim": None,
#     "relax_xlim": None,
#     "combined_xlim": None,
# 
#     "pull_ylim": (40, 85),
#     "relax_ylim": (40, 85),
#     "combined_ylim": (40, 85),
# 
#     "tight_layout": True,
# }
# =============================================================================

# -----------------------------------------------------------------------------
# Binned / averaged APL analysis
# -----------------------------------------------------------------------------
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

    "apl_title": None,
    "slope_title": None,

    "apl_xlabel": "Time (ns)",
    "apl_ylabel": "Average APL (Å²)",
    "slope_xlabel": "Time (ns)",
    "slope_ylabel": "ΔAPL per bin",

    "apl_color": "black",
    "slope_color": "red",

    "pull_color": "#554768",
    "relax_color": "#81B0CD",

    "show_grid": True,
    "show_apl_markers": True,
    "show_slope_markers": True,
    "apl_marker_size": 7.0,
    "slope_marker_size": 7.0,
    "annotation_font_size": 12,

    "show_phase_boundary": True,
    "phase_boundary_linestyle": "--",
    "phase_boundary_color": "black",
    "phase_boundary_linewidth": 1.0,

    "show_legend": True,

    "apl_xlim": (0, 200),
    "apl_ylim": (40, 90),

    "slope_xlim": (0, 200),

    "slope_mode": "delta_per_bin",   # "delta_per_bin" or "per_ns"
    "slope_ylim_fixed": (-2.0, 2.0),
    "slope_major_tick": 0.5,
    "slope_minor_tick": 0.25,
}

CORE_APL_CFG = {
    "enabled": False,
}    
    
# Built-in peptide residue masses (average residue masses in peptide chain, Da)
# Mass of chain = sum(residue masses) + H2O for termini
PEPTIDE_RESIDUE_MASS = {
    "A": 71.0788,
    "R": 156.1875,
    "N": 114.1038,
    "D": 115.0886,
    "C": 103.1388,
    "E": 129.1155,
    "Q": 128.1307,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "L": 113.1594,
    "K": 128.1741,
    "M": 131.1926,
    "F": 147.1766,
    "P": 97.1167,
    "S": 87.0782,
    "T": 101.1051,
    "W": 186.2132,
    "Y": 163.1760,
    "V": 99.1326,
}

# Built-in peptoid monomer masses.
# Unknown values can be left as None and will be reported without crashing.
# Fill/override with MASS_OVERRIDE_JSON if needed.
PEPTOID_MONOMER_MASS = {
    "Na": None,
    "Nk": None,
    "Nr": None,
    "Nke": None,
    "Nd": None,
    "Ne": None,
    "Ni": None,
    "Nl": None,
    "Nf": None,
    "Nw": None,
    "Ny": None,
    "Nq": None,
    "Nn": None,
    "Ns": None,
    "Nt": None,
    "Nv": None,
    "Nm": None,
    "Nc": None,
    "Ng": None,
    "Np": None,
}

WATER_MASS_TERMINI = 18.01528


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def log_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def log_kv(label: str, value) -> None:
    print(f"{label:<32} {value}")


def resolve_analysis_targets() -> List[dict]:
    """
    Returns a list of analysis targets.

    Priority:
    1) MANUAL_MODE direct-file or one final system folder
    2) TARGET_DIR_MODE scanning only TARGET_DIRS
    3) full DATA_ROOT scan
    """
    if MANUAL_MODE and TARGET_DIR_MODE:
        raise ValueError("Use either MANUAL_MODE or TARGET_DIR_MODE, not both.")

    if MANUAL_MODE:
        if MANUAL_SYSTEM_DIR is not None:
            system_dir = Path(MANUAL_SYSTEM_DIR)
            if not is_valid_system_folder(system_dir):
                raise ValueError(
                    f"MANUAL_SYSTEM_DIR is not a valid final system folder: {system_dir}"
                )
            return [{"system_dir": system_dir, "manual": False}]

        direct_files = [MANUAL_PULL_GRO, MANUAL_PULL_XTC, MANUAL_RELAX_GRO, MANUAL_RELAX_XTC]
        if all(x is not None for x in direct_files):
            return [{
                "system_dir": None,
                "manual": True,
                "pull_gro": Path(MANUAL_PULL_GRO),
                "pull_xtc": Path(MANUAL_PULL_XTC),
                "relax_gro": Path(MANUAL_RELAX_GRO),
                "relax_xtc": Path(MANUAL_RELAX_XTC),
                "molecule_name": MANUAL_MOLECULE_NAME,
                "system_type": MANUAL_SYSTEM_TYPE,
                "run_number": str(MANUAL_RUN_NUMBER),
            }]

        raise ValueError(
            "MANUAL_MODE=True but no valid manual input was provided. "
            "Use MANUAL_SYSTEM_DIR or all four direct files."
        )

    if TARGET_DIR_MODE:
        system_dirs = scan_target_directories(TARGET_DIRS)
        return [{"system_dir": p, "manual": False} for p in system_dirs]

    return [{"system_dir": p, "manual": False} for p in scan_system_folders(DATA_ROOT)]

def get_manual_system_output_dir(molecule_name: str, system_type: str, run_number: int) -> Path:
    return (
        OUTPUT_ROOT
        / sanitize_filename(molecule_name)
        / sanitize_filename(system_type)
        / f"run_{run_number}"
    )


def parse_system_components(folder_name: str) -> List[Dict[str, object]]:
    """
    Examples
    --------
    AAA_80
        -> [{"name": "AAA", "count": 80}]

    AAA_20|VVV_60
        -> [{"name": "AAA", "count": 20},
            {"name": "VVV", "count": 60}]

    Nr-Nw-Nw_20|AAA_60
        -> [{"name": "Nr-Nw-Nw", "count": 20},
            {"name": "AAA", "count": 60}]
    """
    components = []

    for part in folder_name.split("|"):
        part = part.strip()
        if "_" not in part:
            raise ValueError(f"Component is missing '_count': {part}")

        name, count_str = part.rsplit("_", 1)
        if not name.strip():
            raise ValueError(f"Empty component name in: {part}")

        count = int(count_str)
        components.append({
            "name": name.strip(),
            "count": count,
        })

    return components


def apply_plot_style() -> None:
    if PLOT_CFG["style_use_default_matplotlib"]:
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


def ensure_dirs() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    if path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    try:
        df.to_excel(path.with_suffix(".xlsx"), index=False)
    except Exception as exc:
        print(f"Could not write Excel file for {path}: {exc}")


def load_mass_overrides() -> Tuple[Dict[str, float], Dict[str, float]]:
    peptide_mass = dict(PEPTIDE_RESIDUE_MASS)
    peptoid_mass = dict(PEPTOID_MONOMER_MASS)

    if MASS_OVERRIDE_JSON.exists():
        with open(MASS_OVERRIDE_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)

        peptide_override = payload.get("peptide", {})
        peptoid_override = payload.get("peptoid", {})

        peptide_mass.update(peptide_override)
        peptoid_mass.update(peptoid_override)

    return peptide_mass, peptoid_mass


def detect_component_type(component_name: str) -> str:
    return "peptoid" if "-" in component_name else "peptide"


def detect_molecule_type(molecule_name: str) -> str:
    components = parse_system_components(molecule_name)
    types = {detect_component_type(comp["name"]) for comp in components}

    if len(types) == 1:
        return list(types)[0]

    return "mixed"


def split_system_folder_name(folder_name: str) -> Tuple[str, int]:
    """
    Returns:
        molecule_name: original composition string
        nmol: total molecules across all components
    """
    components = parse_system_components(folder_name)
    molecule_name = folder_name
    nmol = sum(int(comp["count"]) for comp in components)
    return molecule_name, nmol


def parse_run_number(run_folder_name: str) -> str:
    if not run_folder_name.startswith("run_"):
        raise ValueError(f"Run folder does not start with 'run_': {run_folder_name}")

    run_label = run_folder_name.split("_", 1)[1].strip()

    if not run_label:
        raise ValueError(f"Run folder has empty run label: {run_folder_name}")

    return run_label


def tokenize_component(component_name: str, component_type: str) -> List[str]:
    if component_type == "peptoid":
        return [x.strip() for x in component_name.split("-") if x.strip()]
    return list(component_name.strip())


def tokenize_molecule(molecule_name: str, molecule_type: str) -> List[str]:
    components = parse_system_components(molecule_name)
    tokens = []

    for comp in components:
        comp_name = str(comp["name"])
        comp_count = int(comp["count"])
        comp_type = detect_component_type(comp_name)
        comp_tokens = tokenize_component(comp_name, comp_type)

        for _ in range(comp_count):
            tokens.extend(comp_tokens)

    return tokens


def charge_counts(tokens: List[str], molecule_type: str) -> Tuple[int, int, int]:
    peptide_pos = {"K", "R", "H"}
    peptide_neg = {"D", "E"}
    peptoid_pos = {"Nk", "Nr", "Nke"}
    peptoid_neg = {"Nd", "Ne"}

    pos = sum(tok in peptide_pos or tok in peptoid_pos for tok in tokens)
    neg = sum(tok in peptide_neg or tok in peptoid_neg for tok in tokens)

    net = pos - neg
    return pos, neg, net


def molecular_weight(
    tokens: List[str],
    molecule_type: str,
    peptide_mass: Dict[str, float],
    peptoid_mass: Dict[str, float],
) -> Tuple[Optional[float], List[str]]:
    missing = []

    if molecule_type == "peptide":
        masses = []
        for tok in tokens:
            val = peptide_mass.get(tok)
            if val is None:
                missing.append(tok)
            else:
                masses.append(float(val))
        if missing:
            return None, sorted(set(missing))
        return float(sum(masses) + WATER_MASS_TERMINI), []

    masses = []
    for tok in tokens:
        val = peptoid_mass.get(tok)
        if val is None:
            missing.append(tok)
        else:
            masses.append(float(val))

    if missing:
        return None, sorted(set(missing))
    return float(sum(masses) + WATER_MASS_TERMINI), []


def sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
    )


def pretty_bilayer_name(system_type: str) -> str:
    lookup_system_type = canonical_system_type(system_type)

    mapping = {
        "POPC60CHOL40_W_WF": "POPC60:CHOL40",
        "POPE75POPG25_W_WF": "POPE75:POPG25",
        "POPC80POPS20_W_WF": "POPC80:POPS20",
    }

    return mapping.get(lookup_system_type, system_type.replace("_W_WF", ""))


def wrap_title(title: str, width: Optional[int] = None) -> str:
    """
    Wrap long plot titles so they do not run off the edge of the figure.

    Notes
    -----
    This is character-based wrapping, so larger fonts need smaller widths.
    Change PLOT_CFG["title_wrap_width"] if the title still looks too long.
    """
    if width is None:
        width = PLOT_CFG.get("title_wrap_width", 30)

    return "\n".join(
        textwrap.wrap(
            str(title),
            width=int(width),
            break_long_words=False,
            break_on_hyphens=False,
        )
    )

def find_existing_file(system_dir: Path, candidates: List[str]) -> Path:
    for name in candidates:
        p = system_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of these files were found in {system_dir}: {candidates}"
    )


def make_system_key(molecule_name: str, nmol: int, system_type: str, run_number: int) -> str:
    return f"{molecule_name}|{nmol}|{system_type}|{run_number}"


def append_failed(
    failed_records: List[dict],
    molecule_name: str,
    nmol: Optional[int],
    system_type: Optional[str],
    run_number: Optional[int],
    stage: str,
    reason: str,
    system_path: Optional[Path],
    failure_category: str = "",
    relax_end_time_ns: Optional[float] = np.nan,
    remaining_relax_time_ns: Optional[float] = np.nan,
    pull_gro_exists: bool = False,
    pull_xtc_exists: bool = False,
    pull_cpt_exists: bool = False,
    relax_gro_exists: bool = False,
    relax_xtc_exists: bool = False,
    relax_cpt_exists: bool = False,
    restart_mode: str = "",
    recommended_action: str = "",
) -> None:
    failed_records.append({
        "molecule_name": molecule_name,
        "nmol": nmol,
        "system_type": system_type,
        "run_number": run_number,
        "stage": stage,
        "reason": reason,
        "system_path": "" if system_path is None else str(system_path),

        "failure_category": failure_category,
        "relax_end_time_ns": relax_end_time_ns,
        "remaining_relax_time_ns": remaining_relax_time_ns,

        "pull_gro_exists": pull_gro_exists,
        "pull_xtc_exists": pull_xtc_exists,
        "pull_cpt_exists": pull_cpt_exists,

        "relax_gro_exists": relax_gro_exists,
        "relax_xtc_exists": relax_xtc_exists,
        "relax_cpt_exists": relax_cpt_exists,

        "restart_mode": restart_mode,
        "recommended_action": recommended_action,
    })


def classify_failure_for_restart(system_dir: Path) -> dict:
    pull_gro = system_dir / "Pull.gro"
    pull_xtc = system_dir / "Pull.xtc"
    pull_cpt = system_dir / "Pull.cpt"

    relax_gro = system_dir / "Relax.gro"
    relax_xtc = system_dir / "Relax.xtc"
    relax_cpt = system_dir / "Relax.cpt"

    info = {
        "failure_category": "other",
        "relax_end_time_ns": np.nan,
        "remaining_relax_time_ns": np.nan,

        "pull_gro_exists": pull_gro.exists(),
        "pull_xtc_exists": pull_xtc.exists(),
        "pull_cpt_exists": pull_cpt.exists(),

        "relax_gro_exists": relax_gro.exists(),
        "relax_xtc_exists": relax_xtc.exists(),
        "relax_cpt_exists": relax_cpt.exists(),

        "restart_mode": "",
        "recommended_action": "",
    }

    if not pull_gro.exists() or not pull_xtc.exists() or not relax_gro.exists() or not relax_xtc.exists():
        info["failure_category"] = "missing_files"
        info["restart_mode"] = "not_restartable"
        info["recommended_action"] = "restore missing required files before rerun"
        return info

    try:
        u = mda.Universe(str(relax_gro), str(relax_xtc), in_memory=False)
        final_time_ns = float(u.trajectory[-1].time) / 1000.0
        info["relax_end_time_ns"] = final_time_ns
        info["remaining_relax_time_ns"] = max(0.0, TARGET_RELAX_TIME_NS - final_time_ns)

        if final_time_ns + TARGET_RELAX_TOLERANCE_NS < TARGET_RELAX_TIME_NS:
            info["failure_category"] = "relax_not_100ns"

            if relax_cpt.exists():
                info["restart_mode"] = "checkpoint"
                info["recommended_action"] = "continue relax from Relax.cpt to remaining target time"
            else:
                info["restart_mode"] = "fresh_from_last_structure"
                info["recommended_action"] = "start new relax extension from final Relax.gro structure"
        else:
            info["failure_category"] = "other"
            info["restart_mode"] = "none"
            info["recommended_action"] = "check failure reason manually"

    except Exception:
        info["failure_category"] = "trajectory_unreadable"
        info["restart_mode"] = "manual_check"
        info["recommended_action"] = "trajectory exists but could not be read; inspect files manually"

    return info


def get_po4_and_leaflets(universe: mda.Universe, system_type: str):
    po4 = universe.select_atoms(PO4_SELECTION)

    if len(po4) == 0:
        raise ValueError("No PO4 atoms found.")

    lookup_system_type = canonical_system_type(system_type)

    phospholipid_residues = po4.residues
    n_phospholipids = phospholipid_residues.n_residues

    chol_atoms = universe.select_atoms("resname " + " ".join(sorted(CHOL_RESNAMES)))
    n_chol = chol_atoms.residues.n_residues

    n_membrane_molecules = n_phospholipids + n_chol

    if "CHOL" in lookup_system_type and n_chol == 0:
        assumed_fraction = SYSTEM_TYPE_TO_ASSUMED_PHOSPHOLIPID_FRACTION.get(lookup_system_type)
        if assumed_fraction is None or assumed_fraction <= 0 or assumed_fraction > 1:
            raise ValueError(
                f"Could not detect CHOL residues and no valid fallback fraction defined for "
                f"{system_type} / canonical={lookup_system_type}"
            )
        n_membrane_molecules = n_phospholipids / assumed_fraction
        n_chol = n_membrane_molecules - n_phospholipids

    if n_membrane_molecules == 0:
        raise ValueError("No membrane molecules counted.")

    lipids_per_leaflet = n_membrane_molecules / 2.0

    return {
        "po4": po4,
        "n_phospholipids": float(n_phospholipids),
        "n_chol": float(n_chol),
        "n_membrane_molecules": float(n_membrane_molecules),
        "lipids_per_leaflet": float(lipids_per_leaflet),
    }


def analyze_trajectory(
    gro_path: Path,
    xtc_path: Path,
    phase_name: str,
    system_type: str,
) -> pd.DataFrame:
    """
    Returns one DataFrame with all needed time-series data for this phase.
    Opens the trajectory once only.
    """
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
            raise ValueError(f"{phase_name}: invalid box dimensions encountered: {dims}")

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
        
            # Canonical APL for this project = box-area method.
            "APL": apl_box,
        
            # Retained only for optional future reference.
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


def plot_single_phase(
    df: pd.DataFrame,
    output_png: Path,
    title: str,
    y_col: str,
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    line_color: str,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_CFG["figure_size"], dpi=PLOT_CFG["dpi"])
    ax.plot(df["time_ns"], df[y_col], linewidth=PLOT_CFG["line_width"], color=line_color)
    if PLOT_CFG.get("show_titles", True):
        ax.set_title(
            wrap_title(title, width=PLOT_CFG["title_wrap_width"]),
            pad=PLOT_CFG["title_pad"],
        )
    ax.set_xlabel(PLOT_CFG["x_label"])
    ax.set_ylabel(PLOT_CFG["y_label_box"] if y_col == "APL_BOX" else PLOT_CFG["y_label_po4"])

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if PLOT_CFG["tight_layout"]:
        fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def plot_combined(
    pull_df: pd.DataFrame,
    relax_df: pd.DataFrame,
    output_png: Path,
) -> None:
    apply_plot_style()

    combined_metric = "APL"

    pull_plot = pull_df.copy()
    relax_plot = relax_df.copy()

    pull_end = float(pull_plot["time_ns"].max())
    relax_plot["time_ns_combined"] = relax_plot["time_ns"] + pull_end
    pull_plot["time_ns_combined"] = pull_plot["time_ns"]

    fig, ax = plt.subplots(figsize=PLOT_CFG["figure_size"], dpi=PLOT_CFG["dpi"])
    ax.plot(
        pull_plot["time_ns_combined"],
        pull_plot[combined_metric],
        linewidth=PLOT_CFG["line_width"],
        color=PLOT_CFG["pull_color"],
        label="Pull",
    )
    ax.plot(
        relax_plot["time_ns_combined"],
        relax_plot[combined_metric],
        linewidth=PLOT_CFG["line_width"],
        color=PLOT_CFG["relax_color"],
        label="Relax",
    )

    if PLOT_CFG.get("show_titles", True):
        ax.set_title(
            wrap_title(PLOT_CFG["combined_title"], width=PLOT_CFG["title_wrap_width"]),
            pad=PLOT_CFG["title_pad"],
        )
    ax.set_xlabel(PLOT_CFG["x_label"])
    ax.set_ylabel(PLOT_CFG["y_label_box"] if combined_metric == "APL_BOX" else PLOT_CFG["y_label_po4"])
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


def save_system_timeseries(
    pull_df: pd.DataFrame,
    relax_df: pd.DataFrame,
    output_csv: Path,
) -> None:
    pull_out = pull_df.copy()
    relax_out = relax_df.copy()

    pull_end = float(pull_out["time_ns"].max())
    pull_out["time_ns_combined"] = pull_out["time_ns"]
    relax_out["time_ns_combined"] = relax_out["time_ns"] + pull_end

    out = pd.concat([pull_out, relax_out], ignore_index=True)
    safe_write_csv(out, output_csv)


def image_outputs_exist(system_output_dir: Path) -> bool:
    required = [
        system_output_dir / "pull_APL_vs_time.png",
        system_output_dir / "relax_APL_vs_time.png",
        system_output_dir / "combined_APL_over_time.png",
        system_output_dir / "APL_time_series.csv",
    ]
    return all(p.exists() for p in required)


def binned_outputs_exist(system_output_dir: Path) -> bool:
    required = [
        system_output_dir / "binned_APL.csv",
        system_output_dir / "binned_APL_slope.csv",
    ]

    if BINNED_APL_CFG["plot_enabled"]:
        required.extend([
            system_output_dir / "binned_APL.png",
            system_output_dir / "binned_APL_slope.png",
        ])

    return all(p.exists() for p in required)


def stitch_images_horizontally(image_paths: List[Path], output_path: Path, background=(255, 255, 255)) -> None:
    valid_paths = [p for p in image_paths if p.exists()]
    if not valid_paths:
        return

    images = [plt.imread(str(p)) for p in valid_paths]

    pil_images = []
    from PIL import Image

    for arr in images:
        if arr.dtype != np.uint8:
            arr = (255 * arr).clip(0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(arr))

    total_width = sum(img.size[0] for img in pil_images)
    max_height = max(img.size[1] for img in pil_images)

    stitched = Image.new("RGB", (total_width, max_height), background)

    x = 0
    for img in pil_images:
        y = (max_height - img.size[1]) // 2
        stitched.paste(img, (x, y))
        x += img.size[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(output_path, "PNG")


def ordered_bilayers(bilayers: List[str]) -> List[str]:
    ordered = [b for b in BILAYER_PRIORITY if b in bilayers]
    ordered.extend(sorted([b for b in bilayers if b not in BILAYER_PRIORITY]))
    return ordered


def ordered_bilayer_pairs(existing_bilayers: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    present = [b for b in BILAYER_PRIORITY if b in existing_bilayers]

    if "POPC60CHOL40_W_WF" in present and "POPC80POPS20_W_WF" in present:
        pairs.append(("POPC80POPS20_W_WF", "POPC60CHOL40_W_WF"))
    if "POPC60CHOL40_W_WF" in present and "POPE75POPG25_W_WF" in present:
        pairs.append(("POPE75POPG25_W_WF", "POPC60CHOL40_W_WF"))
    if "POPC80POPS20_W_WF" in present and "POPE75POPG25_W_WF" in present:
        pairs.append(("POPE75POPG25_W_WF", "POPC80POPS20_W_WF"))

    return pairs


# =============================================================================
# NEW: BINNED / AVERAGED APL FUNCTIONS
# =============================================================================

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
            "Mixed"
        )
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


def get_binned_titles(system_label: str) -> Tuple[str, str]:
    metric = BINNED_APL_CFG["metric"]
    bin_size_ns = BINNED_APL_CFG["bin_size_ns"]

    if BINNED_APL_CFG["apl_title"] is None:
        apl_title = f"{system_label} | Average {metric} per {bin_size_ns:g} ns bin"
    else:
        apl_title = BINNED_APL_CFG["apl_title"]

    if BINNED_APL_CFG["slope_title"] is None:
        if BINNED_APL_CFG["slope_mode"] == "delta_per_bin":
            slope_title = f"{system_label} | ΔAPL per {bin_size_ns:g} ns bin"
        else:
            slope_title = f"{system_label} | {metric} slope (per ns)"
    else:
        slope_title = BINNED_APL_CFG["slope_title"]

    return apl_title, slope_title


def plot_binned_apl(
    binned_df: pd.DataFrame,
    output_png: Path,
    system_label: str,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_CFG["figure_size"], dpi=PLOT_CFG["dpi"])

    apl_title, _ = get_binned_titles(system_label)

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

    if PLOT_CFG.get("show_titles", True):
        ax.set_title(
            wrap_title(apl_title, width=PLOT_CFG["title_wrap_width"]),
            pad=PLOT_CFG["title_pad"],
        )

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

    _, slope_title = get_binned_titles(system_label)

    if BINNED_APL_CFG["slope_mode"] == "delta_per_bin":
        y_col = "delta_apl_per_bin"
        ylabel = f"ΔAPL per {BINNED_APL_CFG['bin_size_ns']:g} ns bin"
    else:
        y_col = "slope_apl_per_ns"
        ylabel = "Slope (APL per ns)"

    if slope_df.empty:
        if PLOT_CFG.get("show_titles", True):
            ax.set_title(
                wrap_title(slope_title, width=PLOT_CFG["title_wrap_width"]),
                pad=PLOT_CFG["title_pad"],
            )
        ax.set_xlabel(BINNED_APL_CFG["slope_xlabel"])
        ax.set_ylabel(ylabel)

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

        if BINNED_APL_CFG["show_grid"]:
            ax.grid(True, alpha=0.3)
            if minor is not None:
                ax.grid(True, which="minor", alpha=0.15)

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

    if PLOT_CFG.get("show_titles", True):
        ax.set_title(
            wrap_title(slope_title, width=PLOT_CFG["title_wrap_width"]),
            pad=PLOT_CFG["title_pad"],
        )
    ax.set_xlabel(BINNED_APL_CFG["slope_xlabel"])
    ax.set_ylabel(ylabel)

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


def save_binned_outputs(
    combined_df: pd.DataFrame,
    system_output_dir: Path,
    system_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric = BINNED_APL_CFG["metric"]

    windowed_df = apply_combined_time_window(
        combined_df,
        start_ns=BINNED_APL_CFG["start_ns"],
        end_ns=BINNED_APL_CFG["end_ns"],
    )

    if windowed_df.empty:
        raise ValueError("Combined dataframe became empty after applying binned-analysis time window.")

    binned_df = bin_apl_timeseries(
        df=windowed_df,
        value_col=metric,
        bin_size_ns=float(BINNED_APL_CFG["bin_size_ns"]),
        use_bin_center_for_x=bool(BINNED_APL_CFG["use_bin_center_for_x"]),
        drop_incomplete_last_bin=bool(BINNED_APL_CFG["drop_incomplete_last_bin"]),
    )

    slope_df = compute_binned_slope(binned_df)

    safe_write_csv(binned_df, system_output_dir / "binned_APL.csv")
    safe_write_csv(slope_df, system_output_dir / "binned_APL_slope.csv")

    if BINNED_APL_CFG["plot_enabled"]:
        plot_binned_apl(
            binned_df=binned_df,
            output_png=system_output_dir / "binned_APL.png",
            system_label=system_label,
        )
        plot_binned_slope(
            slope_df=slope_df,
            output_png=system_output_dir / "binned_APL_slope.png",
            system_label=system_label,
        )

    return binned_df, slope_df


# =============================================================================
# GROUPED OUTPUTS
# =============================================================================

def build_grouped_outputs(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates grouped per-molecule CSVs and the bilayer comparison CSV.
    Returns comparison dataframe.
    """
    comparison_rows = []

    if master_df.empty:
        return pd.DataFrame()

    for molecule_name, mol_df in master_df.groupby("molecule_name", sort=True):
        molecule_dir = OUTPUT_ROOT / sanitize_filename(molecule_name)
        molecule_dir.mkdir(parents=True, exist_ok=True)

        mol_df_sorted = mol_df.sort_values(["run_number", "system_type", "nmol"]).reset_index(drop=True)
        safe_write_csv(mol_df_sorted, molecule_dir / "master_rows_for_this_molecule.csv")

        for (run_number, nmol), sub in mol_df.groupby(["run_number", "nmol"], dropna=False):
            bilayers = list(sub["system_type"].unique())
            for subject_bilayer, comparison_bilayer in ordered_bilayer_pairs(bilayers):
                subject = sub[sub["system_type"] == subject_bilayer]
                comparison = sub[sub["system_type"] == comparison_bilayer]

                if subject.empty or comparison.empty:
                    continue

                subject_row = subject.iloc[0]
                comparison_row = comparison.iloc[0]

                comparison_rows.append({
                    "molecule_name": molecule_name,
                    "molecule_type": subject_row["molecule_type"],
                    "run_number": str(run_number),
                    "nmol": int(nmol),

                    "subject_bilayer": subject_bilayer,
                    "comparison_bilayer": comparison_bilayer,

                    "finish_APL_PO4_subject": subject_row["finish_APL_PO4_100ns"],
                    "finish_APL_PO4_comparison": comparison_row["finish_APL_PO4_100ns"],
                    "finish_APL_PO4_difference_subject_minus_comparison":
                        subject_row["finish_APL_PO4_100ns"] - comparison_row["finish_APL_PO4_100ns"],

                    "finish_APL_BOX_subject": subject_row["finish_APL_BOX_100ns"],
                    "finish_APL_BOX_comparison": comparison_row["finish_APL_BOX_100ns"],
                    "finish_APL_BOX_difference_subject_minus_comparison":
                        subject_row["finish_APL_BOX_100ns"] - comparison_row["finish_APL_BOX_100ns"],

                    "start_APL_PO4_subject": subject_row["start_APL_PO4"],
                    "start_APL_PO4_comparison": comparison_row["start_APL_PO4"],
                    "finish_APL_PO4_diff_normalized_to_subject_start_percent":
                        ((subject_row["finish_APL_PO4_100ns"] - comparison_row["finish_APL_PO4_100ns"]) / subject_row["start_APL_PO4"] * 100.0)
                        if pd.notna(subject_row["start_APL_PO4"]) and subject_row["start_APL_PO4"] != 0 else np.nan,
                    "finish_APL_PO4_diff_normalized_to_comparison_start_percent":
                        ((subject_row["finish_APL_PO4_100ns"] - comparison_row["finish_APL_PO4_100ns"]) / comparison_row["start_APL_PO4"] * 100.0)
                        if pd.notna(comparison_row["start_APL_PO4"]) and comparison_row["start_APL_PO4"] != 0 else np.nan,

                    "start_APL_BOX_subject": subject_row["start_APL_BOX"],
                    "start_APL_BOX_comparison": comparison_row["start_APL_BOX"],
                    "finish_APL_BOX_diff_normalized_to_subject_start_percent":
                        ((subject_row["finish_APL_BOX_100ns"] - comparison_row["finish_APL_BOX_100ns"]) / subject_row["start_APL_BOX"] * 100.0)
                        if pd.notna(subject_row["start_APL_BOX"]) and subject_row["start_APL_BOX"] != 0 else np.nan,
                    "finish_APL_BOX_diff_normalized_to_comparison_start_percent":
                        ((subject_row["finish_APL_BOX_100ns"] - comparison_row["finish_APL_BOX_100ns"]) / comparison_row["start_APL_BOX"] * 100.0)
                        if pd.notna(comparison_row["start_APL_BOX"]) and comparison_row["start_APL_BOX"] != 0 else np.nan,
                })

    comparison_df = pd.DataFrame(comparison_rows)

    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(
            ["molecule_name", "run_number", "nmol", "subject_bilayer", "comparison_bilayer"]
        ).reset_index(drop=True)

        for molecule_name, sub in comparison_df.groupby("molecule_name", sort=True):
            molecule_dir = OUTPUT_ROOT / sanitize_filename(molecule_name)
            molecule_dir.mkdir(parents=True, exist_ok=True)
            safe_write_csv(sub, molecule_dir / "bilayercomparison_for_this_molecule.csv")

    return comparison_df


def build_grouped_apl_stacks(master_df: pd.DataFrame) -> None:
    """
    For each molecule + run + nmol, stitch combined APL graphs horizontally.

    Outputs generated:
    1) Normal all-bilayer image:
       stacked_combined_APL_run_<run>_nmol_<nmol>.png

    2) Extra selected two-bilayer image(s), controlled by TWO_BILAYER_STACKS:
       stacked_combined_APL_2bilayers_<pair>_run_<run>_nmol_<nmol>.png
    """
    if master_df.empty:
        return

    def get_combined_pngs_for_bilayers(
        molecule_name: str,
        run_number,
        bilayers: List[str],
    ) -> List[Path]:
        image_paths = []

        for bilayer in bilayers:
            system_output_dir = (
                OUTPUT_ROOT
                / sanitize_filename(molecule_name)
                / sanitize_filename(bilayer)
                / f"run_{run_number}"
            )

            combined_png = system_output_dir / "combined_APL_over_time.png"
            if combined_png.exists():
                image_paths.append(combined_png)

        return image_paths

    for (molecule_name, run_number, nmol), sub in master_df.groupby(
        ["molecule_name", "run_number", "nmol"],
        dropna=False,
    ):
        all_bilayers = ordered_bilayers(list(sub["system_type"].unique()))
        stacked_dir = OUTPUT_ROOT / sanitize_filename(molecule_name)
        stacked_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # 1) Existing all-bilayer combined APL image
        # ---------------------------------------------------------------------
        all_image_paths = get_combined_pngs_for_bilayers(
            molecule_name=molecule_name,
            run_number=run_number,
            bilayers=all_bilayers,
        )

        if all_image_paths:
            stacked_path = stacked_dir / f"stacked_combined_APL_run_{run_number}_nmol_{int(nmol)}.png"
            if not stacked_path.exists():
                stitch_images_horizontally(all_image_paths, stacked_path)

        # ---------------------------------------------------------------------
        # 2) New selected two-bilayer combined APL image(s)
        # ---------------------------------------------------------------------
        if not GENERATE_TWO_BILAYER_STACKS:
            continue

        for bilayer_a, bilayer_b in TWO_BILAYER_STACKS:
            selected_bilayers = [bilayer_a, bilayer_b]

            # Only generate if both bilayers are present for this molecule/run/nmol.
            if not all(b in all_bilayers for b in selected_bilayers):
                continue

            two_image_paths = get_combined_pngs_for_bilayers(
                molecule_name=molecule_name,
                run_number=run_number,
                bilayers=selected_bilayers,
            )

            if len(two_image_paths) != 2:
                continue

            pair_label = "__".join(
                sanitize_filename(pretty_bilayer_name(b)) for b in selected_bilayers
            )

            two_stacked_path = (
                stacked_dir
                / f"stacked_combined_APL_2bilayers_{pair_label}_run_{run_number}_nmol_{int(nmol)}.png"
            )

            if not two_stacked_path.exists():
                stitch_images_horizontally(two_image_paths, two_stacked_path)


def build_grouped_binned_apl_stacks(master_df: pd.DataFrame) -> None:
    """
    For each molecule + run + nmol, create vertical stacked binned_APL plots.

    Outputs generated:
    1) Normal all-bilayer image:
       stacked_binned_APL_run_<run>_nmol_<nmol>.png

    2) Extra selected two-bilayer image(s), controlled by TWO_BILAYER_STACKS:
       stacked_binned_APL_2bilayers_<pair>_run_<run>_nmol_<nmol>.png
    """
    if master_df.empty:
        return

    def load_plot_items_for_bilayers(
        molecule_name: str,
        run_number,
        bilayers: List[str],
    ) -> List[Tuple[str, pd.DataFrame]]:
        plot_items = []

        for bilayer in bilayers:
            system_output_dir = (
                OUTPUT_ROOT
                / sanitize_filename(molecule_name)
                / sanitize_filename(bilayer)
                / f"run_{run_number}"
            )

            binned_csv = system_output_dir / "binned_APL.csv"
            if not binned_csv.exists():
                continue

            try:
                binned_df = pd.read_csv(binned_csv)
            except Exception:
                continue

            needed = {"x_ns", "apl_mean", "phase_label"}
            if binned_df.empty or not needed.issubset(set(binned_df.columns)):
                continue

            plot_items.append((bilayer, binned_df))

        return plot_items

    def save_stacked_binned_apl_plot(
        molecule_name: str,
        run_number,
        nmol,
        plot_items: List[Tuple[str, pd.DataFrame]],
        output_png: Path,
        title_suffix: str,
    ) -> None:
        if not plot_items:
            return

        apply_plot_style()

        nrows = len(plot_items)
        fig_height = max(3.0, 3.6 * nrows)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(PLOT_CFG["figure_size"][0], fig_height),
            dpi=PLOT_CFG["dpi"],
            sharex=True,
            sharey=True,
        )

        if nrows == 1:
            axes = [axes]

        marker = "o" if BINNED_APL_CFG["show_apl_markers"] else None

        for ax, (bilayer, binned_df) in zip(axes, plot_items):
            x = binned_df["x_ns"].to_numpy()
            y = binned_df["apl_mean"].to_numpy()
            phase = binned_df["phase_label"].to_numpy()

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

            if PLOT_CFG.get("show_titles", True):
                ax.set_title(
                    wrap_title(
                        pretty_bilayer_name(bilayer),
                        width=PLOT_CFG["subplot_title_wrap_width"],
                    ),
                    pad=PLOT_CFG["title_pad"],
                )
            ax.set_ylabel(BINNED_APL_CFG["apl_ylabel"])

            if BINNED_APL_CFG["apl_xlim"] is not None:
                ax.set_xlim(*BINNED_APL_CFG["apl_xlim"])
            if BINNED_APL_CFG["apl_ylim"] is not None:
                ax.set_ylim(*BINNED_APL_CFG["apl_ylim"])

            if BINNED_APL_CFG["show_legend"]:
                ax.legend()

        axes[-1].set_xlabel(BINNED_APL_CFG["apl_xlabel"])

        if PLOT_CFG.get("show_titles", True):
            fig.suptitle(
                wrap_title(
                    f"{molecule_name} | run_{run_number} | nmol={int(nmol)} | {title_suffix}",
                    width=PLOT_CFG["suptitle_wrap_width"],
                ),
                fontsize=PLOT_CFG["title_size"],
                y=PLOT_CFG["suptitle_y"],
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            fig.tight_layout()

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png)
        plt.close(fig)

    for (molecule_name, run_number, nmol), sub in master_df.groupby(
        ["molecule_name", "run_number", "nmol"],
        dropna=False,
    ):
        all_bilayers = ordered_bilayers(list(sub["system_type"].unique()))
        stacked_dir = OUTPUT_ROOT / sanitize_filename(molecule_name)
        stacked_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # 1) Existing all-bilayer stacked binned APL image
        # ---------------------------------------------------------------------
        all_plot_items = load_plot_items_for_bilayers(
            molecule_name=molecule_name,
            run_number=run_number,
            bilayers=all_bilayers,
        )

        all_output_png = stacked_dir / f"stacked_binned_APL_run_{run_number}_nmol_{int(nmol)}.png"

        save_stacked_binned_apl_plot(
            molecule_name=molecule_name,
            run_number=run_number,
            nmol=nmol,
            plot_items=all_plot_items,
            output_png=all_output_png,
            title_suffix="stacked binned APL",
        )

        # ---------------------------------------------------------------------
        # 2) New selected two-bilayer stacked binned APL image(s)
        # ---------------------------------------------------------------------
        if not GENERATE_TWO_BILAYER_STACKS:
            continue

        for bilayer_a, bilayer_b in TWO_BILAYER_STACKS:
            selected_bilayers = [bilayer_a, bilayer_b]

            # Only generate if both bilayers are present for this molecule/run/nmol.
            if not all(b in all_bilayers for b in selected_bilayers):
                continue

            two_plot_items = load_plot_items_for_bilayers(
                molecule_name=molecule_name,
                run_number=run_number,
                bilayers=selected_bilayers,
            )

            if len(two_plot_items) != 2:
                continue

            pair_label = "__".join(
                sanitize_filename(pretty_bilayer_name(b)) for b in selected_bilayers
            )

            two_output_png = (
                stacked_dir
                / f"stacked_binned_APL_2bilayers_{pair_label}_run_{run_number}_nmol_{int(nmol)}.png"
            )

            save_stacked_binned_apl_plot(
                molecule_name=molecule_name,
                run_number=run_number,
                nmol=nmol,
                plot_items=two_plot_items,
                output_png=two_output_png,
                title_suffix=(
                    f"stacked binned APL | "
                    f"{pretty_bilayer_name(bilayer_a)} vs {pretty_bilayer_name(bilayer_b)}"
                ),
            )
        



def build_grouped_binned_apl_overlay_plots(master_df: pd.DataFrame) -> None:
    """
    For each molecule + run + nmol, create overlaid binned_APL plots.

    These are different from the stacked plots:
        - stacked plots put each bilayer in its own subplot
        - overlay plots put multiple bilayers on the SAME axis

    Important visual rules for the overlay plot:
        - NO "Mixed" phase is shown
        - colour = phase (Pull / Relax)
        - marker = bilayer
        - legend shows only:
            1) phase colours
            2) bilayer markers
    """
    if master_df.empty:
        return

    if not GENERATE_OVERLAY_BILAYER_GRAPHS:
        return

    def load_plot_items_for_bilayers(
        molecule_name: str,
        run_number,
        bilayers: List[str],
    ) -> List[Tuple[str, pd.DataFrame]]:
        plot_items = []

        for bilayer in bilayers:
            system_output_dir = (
                OUTPUT_ROOT
                / sanitize_filename(molecule_name)
                / sanitize_filename(bilayer)
                / f"run_{run_number}"
            )

            binned_csv = system_output_dir / "binned_APL.csv"
            if not binned_csv.exists():
                continue

            try:
                binned_df = pd.read_csv(binned_csv)
            except Exception:
                continue

            needed = {"x_ns", "apl_mean", "phase_label"}
            if binned_df.empty or not needed.issubset(set(binned_df.columns)):
                continue

            plot_items.append((bilayer, binned_df))

        return plot_items

    def get_phase_colour(phase_name: str) -> str:
        return OVERLAY_PHASE_COLORS.get(phase_name, "black")

    def get_bilayer_marker(bilayer: str, fallback_index: int) -> str:
        if bilayer in OVERLAY_BILAYER_MARKERS:
            return OVERLAY_BILAYER_MARKERS[bilayer]

        fallback_markers = ["o", "s", "^", "D", "v", "P", "X"]
        return fallback_markers[fallback_index % len(fallback_markers)]

    def make_ticks_including_upper_limit(vmin: float, vmax: float, step: float) -> np.ndarray:
        if step <= 0:
            return np.array([vmin, vmax], dtype=float)

        ticks = list(np.arange(vmin, vmax + (step * 0.5), step))

        if len(ticks) == 0:
            ticks = [vmin]

        if abs(ticks[-1] - vmax) > 1e-9:
            ticks.append(vmax)
        else:
            ticks[-1] = vmax

        return np.array(ticks, dtype=float)

    def set_overlay_ticks(ax, xlim, ylim) -> None:
        if xlim is not None:
            xmin, xmax = map(float, xlim)
            ax.set_xlim(xmin, xmax)
            ax.set_xticks(
                make_ticks_including_upper_limit(
                    xmin,
                    xmax,
                    float(OVERLAY_X_TICK_STEP),
                )
            )
            ax.set_xlim(xmin, xmax)

        if ylim is not None:
            ymin, ymax = map(float, ylim)
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(
                make_ticks_including_upper_limit(
                    ymin,
                    ymax,
                    float(OVERLAY_Y_TICK_STEP),
                )
            )
            ax.set_ylim(ymin, ymax)

    def plot_phase_segment(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        phase: np.ndarray,
        phase_name: str,
        bilayer: str,
        bilayer_index: int,
    ) -> None:
        # Mixed points are intentionally not plotted.
        if phase_name == "Mixed":
            return

        phase_idx = np.where(phase == phase_name)[0]

        if len(phase_idx) == 0:
            return

        plot_idx = phase_idx

        # For Relax only, include the previous point so the trace joins cleanly.
        if phase_name == "Relax" and phase_idx[0] > 0:
            plot_idx = np.concatenate(([phase_idx[0] - 1], phase_idx))

        phase_colour = get_phase_colour(phase_name)

        ax.plot(
            x[plot_idx],
            y[plot_idx],
            linewidth=PLOT_CFG["line_width"],
            color=phase_colour,
            linestyle="-",
            marker=get_bilayer_marker(bilayer, bilayer_index),
            markersize=BINNED_APL_CFG["apl_marker_size"],
            markerfacecolor=phase_colour,
            markeredgecolor=phase_colour,
            label=None,
            zorder=3,
        )

    def make_overlay_legend(ax, plot_items: List[Tuple[str, pd.DataFrame]]) -> None:
        if not BINNED_APL_CFG["show_legend"]:
            return

        phase_handles = [
            Line2D(
                [0], [0],
                color=get_phase_colour("Pull"),
                linestyle="-",
                linewidth=PLOT_CFG["line_width"],
                marker=None,
                label="Pull",
            ),
            Line2D(
                [0], [0],
                color=get_phase_colour("Relax"),
                linestyle="-",
                linewidth=PLOT_CFG["line_width"],
                marker=None,
                label="Relax",
            ),
        ]

        bilayer_handles = []
        for idx, (bilayer, _) in enumerate(plot_items):
            bilayer_handles.append(
                Line2D(
                    [0], [0],
                    color="black",
                    linestyle="None",
                    marker=get_bilayer_marker(bilayer, idx),
                    markersize=BINNED_APL_CFG["apl_marker_size"],
                    label=pretty_bilayer_name(bilayer),
                )
            )

        handles = phase_handles + bilayer_handles

        # Forced inside the plotting axes:
        #   loc="upper right" anchors the legend's upper-right corner
        #   bbox_to_anchor=(0.98, 0.98) places that corner just inside the axes
        #   bbox_transform=ax.transAxes makes the coordinates refer to the axes,
        #   not the whole figure canvas.
        ax.legend(
            handles=handles,
            loc=OVERLAY_LEGEND_LOC,
            bbox_to_anchor=OVERLAY_LEGEND_BBOX_TO_ANCHOR,
            bbox_transform=ax.transAxes,
            fontsize=OVERLAY_LEGEND_FONT_SIZE,
            frameon=True,
            framealpha=OVERLAY_LEGEND_FRAME_ALPHA,
            borderpad=0.35,
            labelspacing=0.30,
            handletextpad=0.55,
            handlelength=1.50,
        )

    def save_overlay_binned_apl_plot(
        molecule_name: str,
        run_number,
        nmol,
        plot_items: List[Tuple[str, pd.DataFrame]],
        output_png: Path,
        title_suffix: str,
    ) -> None:
        if len(plot_items) < 2:
            return

        apply_plot_style()

        fig, ax = plt.subplots(
            figsize=OVERLAY_FIGURE_SIZE,
            dpi=PLOT_CFG["dpi"],
        )

        for idx, (bilayer, binned_df) in enumerate(plot_items):
            x = binned_df["x_ns"].to_numpy()
            y = binned_df["apl_mean"].to_numpy()
            phase = binned_df["phase_label"].to_numpy()

            plot_phase_segment(
                ax=ax,
                x=x,
                y=y,
                phase=phase,
                phase_name="Pull",
                bilayer=bilayer,
                bilayer_index=idx,
            )

            plot_phase_segment(
                ax=ax,
                x=x,
                y=y,
                phase=phase,
                phase_name="Relax",
                bilayer=bilayer,
                bilayer_index=idx,
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

        title = f"{molecule_name} | run_{run_number} | nmol={int(nmol)} | {title_suffix}"

        if OVERLAY_SHOW_TITLE:
            ax.set_title(
                wrap_title(title, width=PLOT_CFG["title_wrap_width"]),
                fontsize=OVERLAY_TITLE_FONT_SIZE,
                pad=OVERLAY_TITLE_PAD,
                y=OVERLAY_TITLE_Y,
            )

        ax.set_xlabel(BINNED_APL_CFG["apl_xlabel"])
        ax.set_ylabel(BINNED_APL_CFG["apl_ylabel"])

        set_overlay_ticks(
            ax=ax,
            xlim=BINNED_APL_CFG["apl_xlim"],
            ylim=BINNED_APL_CFG["apl_ylim"],
        )

        make_overlay_legend(ax, plot_items)

        if PLOT_CFG["tight_layout"]:
            fig.tight_layout()

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png)
        plt.close(fig)

    for (molecule_name, run_number, nmol), sub in master_df.groupby(
        ["molecule_name", "run_number", "nmol"],
        dropna=False,
    ):
        all_bilayers = ordered_bilayers(list(sub["system_type"].unique()))
        overlay_dir = OUTPUT_ROOT / sanitize_filename(molecule_name)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        safe_molecule_name = sanitize_filename(str(molecule_name))

        # ---------------------------------------------------------------------
        # 1) Optional all-bilayer overlay
        # ---------------------------------------------------------------------
        if GENERATE_ALL_BILAYER_OVERLAY:
            all_plot_items = load_plot_items_for_bilayers(
                molecule_name=molecule_name,
                run_number=run_number,
                bilayers=all_bilayers,
            )

            all_output_png = (
                overlay_dir
                / OVERLAY_ALL_BILAYERS_FILENAME_TEMPLATE.format(
                    molecule_name=safe_molecule_name,
                    run_number=run_number,
                    nmol=int(nmol),
                )
            )

            save_overlay_binned_apl_plot(
                molecule_name=molecule_name,
                run_number=run_number,
                nmol=nmol,
                plot_items=all_plot_items,
                output_png=all_output_png,
                title_suffix="overlaid binned APL | all bilayers",
            )

        # ---------------------------------------------------------------------
        # 2) Optional selected two-bilayer overlays
        # ---------------------------------------------------------------------
        if not GENERATE_TWO_BILAYER_OVERLAYS:
            continue

        for bilayer_a, bilayer_b in TWO_BILAYER_STACKS:
            selected_bilayers = [bilayer_a, bilayer_b]

            if not all(b in all_bilayers for b in selected_bilayers):
                continue

            two_plot_items = load_plot_items_for_bilayers(
                molecule_name=molecule_name,
                run_number=run_number,
                bilayers=selected_bilayers,
            )

            if len(two_plot_items) != 2:
                continue

            pair_label = "__".join(
                sanitize_filename(pretty_bilayer_name(b)) for b in selected_bilayers
            )

            two_output_png = (
                overlay_dir
                / OVERLAY_TWO_BILAYERS_FILENAME_TEMPLATE.format(
                    molecule_name=safe_molecule_name,
                    pair_label=pair_label,
                    run_number=run_number,
                    nmol=int(nmol),
                )
            )

            save_overlay_binned_apl_plot(
                molecule_name=molecule_name,
                run_number=run_number,
                nmol=nmol,
                plot_items=two_plot_items,
                output_png=two_output_png,
                title_suffix=(
                    f"overlaid binned APL | "
                    f"{pretty_bilayer_name(bilayer_a)} vs {pretty_bilayer_name(bilayer_b)}"
                ),
            )


REQUIRED_SYSTEM_FILES = [
    "Pull.gro",
    "Pull.xtc",
    "Relax.gro",
    "Relax.xtc",
]


def is_valid_system_folder(system_dir: Path) -> bool:
    if not system_dir.is_dir():
        return False

    if not all((system_dir / name).exists() for name in REQUIRED_SYSTEM_FILES):
        return False

    try:
        parse_system_components(system_dir.name)
    except Exception:
        return False

    try:
        parse_run_number(system_dir.parent.name)
    except Exception:
        return False

    return True


def scan_target_directory(target_dir: Path) -> List[Path]:
    """
    Accepts a final system folder, run folder, bilayer folder, or higher-level folder.
    Returns only final system folders containing all required trajectory files.
    """
    target_dir = Path(target_dir)

    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    if is_valid_system_folder(target_dir):
        return [target_dir]

    folders = []
    for pull_gro in target_dir.rglob("Pull.gro"):
        system_dir = pull_gro.parent
        if is_valid_system_folder(system_dir):
            folders.append(system_dir)

    return sorted(set(folders))


def scan_target_directories(target_dirs: List[str]) -> List[Path]:
    all_folders = []

    for target in target_dirs:
        target_path = Path(target)
        found = scan_target_directory(target_path)

        if not found:
            print(f"No valid systems found under target directory: {target_path}")

        all_folders.extend(found)

    return sorted(set(all_folders))


def scan_system_folders(root: Path) -> List[Path]:
    folders = []
    for bilayer_dir in sorted(root.iterdir()):
        if not bilayer_dir.is_dir():
            continue

        for run_dir in sorted(bilayer_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            for system_dir in sorted(run_dir.iterdir()):
                if not system_dir.is_dir():
                    continue
            
                try:
                    parse_system_components(system_dir.name)
                except Exception:
                    print(f"Skipping non-system folder: {system_dir}")
                    continue
            
                folders.append(system_dir)

    return folders


def get_system_output_dir(molecule_name: str, system_type: str, run_number: int) -> Path:
    return (
        OUTPUT_ROOT
        / sanitize_filename(molecule_name)
        / sanitize_filename(system_type)
        / f"run_{run_number}"
    )


def get_slope_ylim(bin_size_ns: float) -> Tuple[float, float]:
    if BINNED_APL_CFG["slope_mode"] == "delta_per_bin":
        return tuple(BINNED_APL_CFG["slope_ylim_fixed"])

    apl_ymin, apl_ymax = BINNED_APL_CFG["apl_ylim"]
    max_abs = (apl_ymax - apl_ymin) / float(bin_size_ns)
    return (-max_abs, max_abs)


# =============================================================================
# MAIN SYSTEM ANALYSIS
# =============================================================================

def analyze_system(
    system_dir: Optional[Path],
    peptide_mass: Dict[str, float],
    peptoid_mass: Dict[str, float],
    manual_target: Optional[dict] = None,
) -> dict:
    """
    Normal mode:
        system_dir from PFPeptoids folder layout

    Manual direct-file mode:
        system_dir=None
        manual_target contains all file paths + labels
    """
    if manual_target is None:
        system_type = system_dir.parents[1].name
        run_number = parse_run_number(system_dir.parent.name)
        molecule_name, nmol = split_system_folder_name(system_dir.name)

        pull_gro = system_dir / "Pull.gro"
        pull_xtc = system_dir / "Pull.xtc"
        relax_gro = system_dir / "Relax.gro"
        relax_xtc = system_dir / "Relax.xtc"
        system_dir_str = str(system_dir)
    else:
        system_type = manual_target["system_type"]
        run_number = str(manual_target["run_number"])
        molecule_name = str(manual_target["molecule_name"])
        _, nmol = split_system_folder_name(molecule_name)

        pull_gro = Path(manual_target["pull_gro"])
        pull_xtc = Path(manual_target["pull_xtc"])
        relax_gro = Path(manual_target["relax_gro"])
        relax_xtc = Path(manual_target["relax_xtc"])
        system_dir_str = str(pull_gro.parent)

    molecule_type = detect_molecule_type(molecule_name)

    tokens = tokenize_molecule(molecule_name, molecule_type)
    length = len(tokens)
    pos_charge, neg_charge, net_charge = charge_counts(tokens, molecule_type)

    mw, missing_mass_codes = molecular_weight(
        tokens=tokens,
        molecule_type=molecule_type,
        peptide_mass=peptide_mass,
        peptoid_mass=peptoid_mass,
    )

    charge_density = (net_charge / length) if length > 0 else np.nan

    required = [pull_gro, pull_xtc, relax_gro, relax_xtc]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {missing}")

    log_section(f"Analyzing system: {molecule_name}")
    log_kv("System type", system_type)
    log_kv("Run number", run_number)
    log_kv("System source", system_dir_str)

    pull_df = analyze_trajectory(pull_gro, pull_xtc, phase_name="pull", system_type=system_type)
    relax_df = analyze_trajectory(relax_gro, relax_xtc, phase_name="relax", system_type=system_type)

    nearest_100 = nearest_row(relax_df, TARGET_RELAX_TIME_NS)
    nearest_time_ns = float(nearest_100["time_ns"])
    if abs(nearest_time_ns - TARGET_RELAX_TIME_NS) > TARGET_RELAX_TOLERANCE_NS:
        raise ValueError(
            f"Relax trajectory does not reach 100 ns within tolerance. "
            f"Nearest frame = {nearest_time_ns:.3f} ns"
        )

    start_row = pull_df.iloc[0]
    finish_row = nearest_100

    relax_end_ns = float(relax_df["time_ns"].max())
    apl_po4_mean_last10, apl_po4_std_last10, apl_box_mean_last10, apl_box_std_last10 = last_10ns_stats(
        relax_df, relax_end_ns
    )

    start_apl_po4 = float(start_row["APL_PO4"])
    finish_apl_po4 = float(finish_row["APL_PO4"])
    start_apl_box = float(start_row["APL_BOX"])
    finish_apl_box = float(finish_row["APL_BOX"])

    start_thickness = float(start_row["thickness_z_span"])
    finish_thickness = float(finish_row["thickness_z_span"])

    membrane_area_start = float(start_row["membrane_area"])
    membrane_area_finish = float(finish_row["membrane_area"])
    lipids_per_leaflet = float(start_row["lipids_per_leaflet"])
    n_phospholipids = float(start_row["n_phospholipids"])
    n_chol = float(start_row["n_chol"])
    n_membrane_molecules = float(start_row["n_membrane_molecules"])

    system_output_dir = get_system_output_dir(molecule_name, system_type, run_number)
    system_output_dir.mkdir(parents=True, exist_ok=True)

    log_kv("Output directory", system_output_dir)

    single_metric = "APL"
    
    if CORE_APL_CFG["enabled"]:
        if not image_outputs_exist(system_output_dir):
            log_kv("Generating core plots", "yes")
    
            save_system_timeseries(
                pull_df=pull_df,
                relax_df=relax_df,
                output_csv=system_output_dir / "APL_time_series.csv",
            )
    
            plot_single_phase(
                df=pull_df,
                output_png=system_output_dir / "pull_APL_vs_time.png",
                title=PLOT_CFG["pull_title"],
                y_col=single_metric,
                xlim=PLOT_CFG["pull_xlim"],
                ylim=PLOT_CFG["pull_ylim"],
                line_color=PLOT_CFG["pull_color"],
            )
    
            plot_single_phase(
                df=relax_df,
                output_png=system_output_dir / "relax_APL_vs_time.png",
                title=PLOT_CFG["relax_title"],
                y_col=single_metric,
                xlim=PLOT_CFG["relax_xlim"],
                ylim=PLOT_CFG["relax_ylim"],
                line_color=PLOT_CFG["relax_color"],
            )
    
            plot_combined(
                pull_df=pull_df,
                relax_df=relax_df,
                output_png=system_output_dir / "combined_APL_over_time.png",
            )
        else:
            log_kv("Generating core plots", "skipped (already present)")
    else:
        log_kv("Generating core plots", "disabled")

    if BINNED_APL_CFG["enabled"]:
        if not binned_outputs_exist(system_output_dir):
            log_kv("Generating binned outputs", "yes")
            combined_df = combine_pull_relax_timeseries(pull_df, relax_df)
            
            pretty_bilayer = pretty_bilayer_name(system_type)
            
            binned_df, slope_df = save_binned_outputs(
                combined_df=combined_df,
                system_output_dir=system_output_dir,
                system_label=f"{molecule_name} {pretty_bilayer} run_{run_number}",
            )

            log_kv("Binned metric", BINNED_APL_CFG["metric"])
            log_kv("Bin size (ns)", BINNED_APL_CFG["bin_size_ns"])
            log_kv("Binned rows", len(binned_df))
            log_kv("Slope rows", len(slope_df))
        else:
            log_kv("Generating binned outputs", "skipped (already present)")
    else:
        log_kv("Generating binned outputs", "disabled")

    components = parse_system_components(molecule_name)

    component_names = [str(c["name"]) for c in components]
    component_counts = [int(c["count"]) for c in components]
    component_types = [detect_component_type(name) for name in component_names]

    component_token_sequences = []
    component_lengths = []
    component_positive_charges = []
    component_negative_charges = []
    component_net_charges = []
    component_molecular_weights = []
    component_missing_mass_codes = []

    for comp_name, comp_type in zip(component_names, component_types):
        comp_tokens = tokenize_component(comp_name, comp_type)

        comp_pos, comp_neg, comp_net = charge_counts(comp_tokens, comp_type)

        comp_mw, comp_missing = molecular_weight(
            tokens=comp_tokens,
            molecule_type=comp_type,
            peptide_mass=peptide_mass,
            peptoid_mass=peptoid_mass,
        )

        if comp_type == "peptoid":
            component_token_sequences.append("-".join(comp_tokens))
        else:
            component_token_sequences.append("".join(comp_tokens))

        component_lengths.append(len(comp_tokens))
        component_positive_charges.append(comp_pos)
        component_negative_charges.append(comp_neg)
        component_net_charges.append(comp_net)
        component_molecular_weights.append("" if comp_mw is None else f"{comp_mw:.6f}")
        component_missing_mass_codes.append(";".join(comp_missing))

    all_missing_mass_codes = sorted(
        {
            code
            for missing_group in component_missing_mass_codes
            for code in missing_group.split(";")
            if code
        }
    )

    row = {
        "system_key": make_system_key(molecule_name, nmol, system_type, run_number),

        "molecule_name": molecule_name,
        "molecule_type": molecule_type,
        "nmol": nmol,

        "component_count": len(components),
        "component_names": "|".join(component_names),
        "component_counts": "|".join(str(x) for x in component_counts),
        "component_types": "|".join(component_types),
        "component_token_sequences": "|".join(component_token_sequences),
        "component_lengths": "|".join(str(x) for x in component_lengths),
        "component_molecular_weights": "|".join(component_molecular_weights),
        "component_missing_mass_codes": "|".join(component_missing_mass_codes),
        "component_positive_charges": "|".join(str(x) for x in component_positive_charges),
        "component_negative_charges": "|".join(str(x) for x in component_negative_charges),
        "component_net_charges": "|".join(str(x) for x in component_net_charges),
        "contains_mixture": len(components) > 1,

        "length": "|".join(str(x) for x in component_lengths),
        "molecular_weight": "|".join(component_molecular_weights),
        "molecular_weight_missing_codes": ";".join(all_missing_mass_codes),

        "positive_charge": "|".join(str(x) for x in component_positive_charges),
        "negative_charge": "|".join(str(x) for x in component_negative_charges),
        "net_charge": "|".join(str(x) for x in component_net_charges),
        "charge_density": "|".join(
            f"{net / length:.6f}" if length > 0 else ""
            for net, length in zip(component_net_charges, component_lengths)
        ),

        "system_type": system_type,
        "run_number": run_number,

        "start_APL": start_apl_box,
        "finish_APL_100ns": finish_apl_box,
        "APL_change": finish_apl_box - start_apl_box,
        "APL_percent_change": ((finish_apl_box - start_apl_box) / start_apl_box * 100.0) if start_apl_box != 0 else np.nan,
        "APL_mean_last10ns": apl_box_mean_last10,
        "APL_std_last10ns": apl_box_std_last10,
        "APL_method": "BOX",

        "start_APL_PO4": start_apl_po4,
        "finish_APL_PO4_100ns": finish_apl_po4,
        "APL_change_PO4": finish_apl_po4 - start_apl_po4,
        "APL_percent_change_PO4": ((finish_apl_po4 - start_apl_po4) / start_apl_po4 * 100.0) if start_apl_po4 != 0 else np.nan,
        "APL_mean_last10ns_PO4": apl_po4_mean_last10,
        "APL_std_last10ns_PO4": apl_po4_std_last10,

        "start_APL_BOX": start_apl_box,
        "finish_APL_BOX_100ns": finish_apl_box,
        "APL_change_BOX": finish_apl_box - start_apl_box,
        "APL_percent_change_BOX": ((finish_apl_box - start_apl_box) / start_apl_box * 100.0) if start_apl_box != 0 else np.nan,
        "APL_mean_last10ns_BOX": apl_box_mean_last10,
        "APL_std_last10ns_BOX": apl_box_std_last10,

        "start_thickness": start_thickness,
        "finish_thickness_100ns": finish_thickness,
        "thickness_change": start_thickness - finish_thickness,

        "lipids_per_leaflet": lipids_per_leaflet,
        "membrane_area_start": membrane_area_start,
        "membrane_area_finish": membrane_area_finish,

        "final_relax_time_ns": relax_end_ns,
        "timepoint_100ns_found": True,
        "analysis_complete": True,

        "has_missing_mass": len(all_missing_mass_codes) > 0,
        "missing_mass_count": len(all_missing_mass_codes),
        "relax_end_time_ns": relax_end_ns,
        "reached_100ns": abs(nearest_time_ns - TARGET_RELAX_TIME_NS) <= TARGET_RELAX_TOLERANCE_NS,
        "target_relax_time_ns": TARGET_RELAX_TIME_NS,
        "remaining_relax_time_ns": max(0.0, TARGET_RELAX_TIME_NS - relax_end_ns),

        "token_sequence": "|".join(component_token_sequences),

        "system_dir": system_dir_str,
        "output_dir": str(system_output_dir),

        "n_phospholipids": n_phospholipids,
        "n_chol": n_chol,
        "n_membrane_molecules": n_membrane_molecules,
    }

    return row


def write_metadata() -> None:
    metadata = {
        "data_root": str(DATA_ROOT),
        "manual_mode": MANUAL_MODE,
        "target_dir_mode": TARGET_DIR_MODE,
        "target_selection_mode": TARGET_SELECTION_MODE if TARGET_DIR_MODE else None,
        "analysis_run": (
            NORMALIZED_ANALYSIS_RUN
            if TARGET_DIR_MODE and TARGET_SELECTION_MODE == "run"
            else None
        ),
        "analysis_bilayers": list(ANALYSIS_BILAYERS),
        "target_dirs": [str(path) for path in TARGET_DIRS] if TARGET_DIR_MODE else [],
        "master_csv": str(MASTER_CSV),
        "comparison_csv": str(COMPARISON_CSV),
        "failed_csv": str(FAILED_CSV),
        "output_root": str(OUTPUT_ROOT),
        "target_relax_time_ns": TARGET_RELAX_TIME_NS,
        "target_relax_tolerance_ns": TARGET_RELAX_TOLERANCE_NS,
        "po4_selection": PO4_SELECTION,
        "bilayer_priority": BILAYER_PRIORITY,
        "two_bilayer_stacks": TWO_BILAYER_STACKS,
        "generate_two_bilayer_stacks": GENERATE_TWO_BILAYER_STACKS,
        "generate_overlay_bilayer_graphs": GENERATE_OVERLAY_BILAYER_GRAPHS,
        "generate_all_bilayer_overlay": GENERATE_ALL_BILAYER_OVERLAY,
        "generate_two_bilayer_overlays": GENERATE_TWO_BILAYER_OVERLAYS,
        "overlay_all_bilayers_filename_template": OVERLAY_ALL_BILAYERS_FILENAME_TEMPLATE,
        "overlay_two_bilayers_filename_template": OVERLAY_TWO_BILAYERS_FILENAME_TEMPLATE,
        "overlay_show_title": OVERLAY_SHOW_TITLE,
        "overlay_figure_size": OVERLAY_FIGURE_SIZE,
        "overlay_legend_loc": OVERLAY_LEGEND_LOC,
        "overlay_legend_bbox_to_anchor": OVERLAY_LEGEND_BBOX_TO_ANCHOR,
        "overlay_legend_font_size": OVERLAY_LEGEND_FONT_SIZE,
        "overlay_phase_colors": OVERLAY_PHASE_COLORS,
        "overlay_bilayer_markers": OVERLAY_BILAYER_MARKERS,
        "overlay_bilayer_colors": OVERLAY_BILAYER_COLORS,
        "overlay_phase_linestyles": OVERLAY_PHASE_LINESTYLES,
        "overlay_phase_markers": OVERLAY_PHASE_MARKERS,
        "plot_config": PLOT_CFG,
        "binned_apl_config": BINNED_APL_CFG,
        "mass_override_json": str(MASS_OVERRIDE_JSON),
        "molecule_type_rule": {
            "peptide": "no hyphens",
            "peptoid": "hyphen-separated monomers",
        },
        "charge_rules": {
            "peptide_positive": ["K", "R", "H"],
            "peptide_negative": ["D", "E"],
            "peptoid_positive": ["Nk", "Nr", "Nke"],
            "peptoid_negative": ["Nd", "Ne"],
        },
        "apl_methods": {
            "APL_PO4": "(PO4 x-span * PO4 y-span) / lipids_per_leaflet",
            "APL_BOX": "(box_x * box_y) / lipids_per_leaflet",
        },
        "binned_outputs": {
            "binned_APL.csv": "mean/std/min/max/n_frames per time bin",
            "binned_APL_slope.csv": "slope between consecutive binned means",
        },
    }

    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dirs()
    peptide_mass, peptoid_mass = load_mass_overrides()
    write_metadata()

    log_section("APL ANALYSIS CONFIG")
    log_kv("Script path", SCRIPT_PATH)
    log_kv("Analysis directory", ANALYSIS_DIR)
    log_kv("Data root", DATA_ROOT)
    log_kv("Output root", OUTPUT_ROOT)
    log_kv("Manual mode", MANUAL_MODE)
    log_kv("Target dir mode", TARGET_DIR_MODE)
    if TARGET_DIR_MODE:
        log_kv("Target selection mode", TARGET_SELECTION_MODE)
        if TARGET_SELECTION_MODE == "run":
            log_kv("Analysis run", NORMALIZED_ANALYSIS_RUN)
            log_kv("Analysis bilayers", ANALYSIS_BILAYERS)
        log_kv("Target dirs", TARGET_DIRS)
    log_kv("Binned analysis enabled", BINNED_APL_CFG["enabled"])
    if BINNED_APL_CFG["enabled"]:
        log_kv("Binned metric", BINNED_APL_CFG["metric"])
        log_kv("Binned bin size (ns)", BINNED_APL_CFG["bin_size_ns"])

    existing_master = safe_read_csv(MASTER_CSV)
    existing_failed = pd.DataFrame()

    existing_keys = set()
    if not existing_master.empty and "system_key" in existing_master.columns:
        existing_keys = set(existing_master["system_key"].astype(str).tolist())

    new_rows = []
    failed_records = []

    targets = resolve_analysis_targets()
    log_kv("Targets found", len(targets))

    for idx, target in enumerate(targets, start=1):
        system_dir = target.get("system_dir", None)
        manual_target = target if target.get("manual", False) else None

        try:
            if manual_target is None:
                system_type = system_dir.parents[1].name
                run_number = parse_run_number(system_dir.parent.name)
                molecule_name, nmol = split_system_folder_name(system_dir.name)
            else:
                system_type = manual_target["system_type"]
                run_number = str(manual_target["run_number"])
                molecule_name = str(manual_target["molecule_name"])
                _, nmol = split_system_folder_name(molecule_name)

            system_key = make_system_key(molecule_name, nmol, system_type, run_number)
            system_output_dir = get_system_output_dir(molecule_name, system_type, run_number)

            print(f"\n[{idx}/{len(targets)}] {molecule_name} | {system_type} | run_{run_number}")

            row_missing = system_key not in existing_keys
            plots_missing = CORE_APL_CFG["enabled"] and not image_outputs_exist(system_output_dir)
            binned_missing = BINNED_APL_CFG["enabled"] and not binned_outputs_exist(system_output_dir)

            if not row_missing and not plots_missing and not binned_missing:
                print("  -> skipped: master row and all outputs already present")
                continue

            row = analyze_system(
                system_dir=system_dir,
                peptide_mass=peptide_mass,
                peptoid_mass=peptoid_mass,
                manual_target=manual_target,
            )

            if row_missing:
                new_rows.append(row)
                print("  -> master row added")
            else:
                print("  -> master row already existed; outputs refreshed as needed")

        except Exception as exc:
            print(f"  -> failed: {type(exc).__name__}: {exc}")

            try:
                if manual_target is None:
                    system_type = system_dir.parents[1].name
                    run_number = parse_run_number(system_dir.parent.name)
                    molecule_name, nmol = split_system_folder_name(system_dir.name)
                    failure_path = system_dir
                else:
                    system_type = manual_target["system_type"]
                    run_number = str(manual_target["run_number"])
                    molecule_name = str(manual_target["molecule_name"])
                    _, nmol = split_system_folder_name(molecule_name)
                    failure_path = Path(manual_target["pull_gro"]).parent
            except Exception:
                system_type = None
                run_number = None
                molecule_name = "manual_target"
                nmol = None
                failure_path = None

            if system_dir is not None:
                try:
                    failure_info = classify_failure_for_restart(system_dir)
                except Exception:
                    failure_info = {
                        "failure_category": "other",
                        "relax_end_time_ns": np.nan,
                        "remaining_relax_time_ns": np.nan,
                        "pull_gro_exists": False,
                        "pull_xtc_exists": False,
                        "pull_cpt_exists": False,
                        "relax_gro_exists": False,
                        "relax_xtc_exists": False,
                        "relax_cpt_exists": False,
                        "restart_mode": "",
                        "recommended_action": "",
                    }
            else:
                failure_info = {
                    "failure_category": "manual_input_error",
                    "relax_end_time_ns": np.nan,
                    "remaining_relax_time_ns": np.nan,
                    "pull_gro_exists": False,
                    "pull_xtc_exists": False,
                    "pull_cpt_exists": False,
                    "relax_gro_exists": False,
                    "relax_xtc_exists": False,
                    "relax_cpt_exists": False,
                    "restart_mode": "manual_check",
                    "recommended_action": "check provided manual file paths",
                }

            append_failed(
                failed_records=failed_records,
                molecule_name=molecule_name,
                nmol=nmol,
                system_type=system_type,
                run_number=run_number,
                stage="analysis",
                reason=f"{type(exc).__name__}: {exc}",
                system_path=failure_path,
                failure_category=failure_info["failure_category"],
                relax_end_time_ns=failure_info["relax_end_time_ns"],
                remaining_relax_time_ns=failure_info["remaining_relax_time_ns"],
                pull_gro_exists=failure_info["pull_gro_exists"],
                pull_xtc_exists=failure_info["pull_xtc_exists"],
                pull_cpt_exists=failure_info["pull_cpt_exists"],
                relax_gro_exists=failure_info["relax_gro_exists"],
                relax_xtc_exists=failure_info["relax_xtc_exists"],
                relax_cpt_exists=failure_info["relax_cpt_exists"],
                restart_mode=failure_info["restart_mode"],
                recommended_action=failure_info["recommended_action"],
            )

    if new_rows:
        new_master = pd.DataFrame(new_rows)
        master_df = pd.concat([existing_master, new_master], ignore_index=True)
    else:
        master_df = existing_master.copy()

    if not master_df.empty:
        master_df = master_df.drop_duplicates(subset=["system_key"], keep="first")
        master_df = master_df.sort_values(["molecule_name", "system_type", "run_number", "nmol"]).reset_index(drop=True)
    safe_write_csv(master_df, MASTER_CSV)

    failed_df = pd.DataFrame(failed_records)
    if not existing_failed.empty and not failed_df.empty:
        failed_df = pd.concat([existing_failed, failed_df], ignore_index=True)
    elif not existing_failed.empty:
        failed_df = existing_failed.copy()

    if not failed_df.empty:
        failed_df = failed_df.drop_duplicates(
            subset=["molecule_name", "nmol", "system_type", "run_number", "stage", "reason"],
            keep="last",
        ).sort_values(["molecule_name", "system_type", "run_number"], na_position="last").reset_index(drop=True)

    safe_write_csv(failed_df, FAILED_CSV)

    comparison_df = build_grouped_outputs(master_df)
    safe_write_csv(comparison_df, COMPARISON_CSV)

    build_grouped_apl_stacks(master_df)
    build_grouped_binned_apl_stacks(master_df)
    build_grouped_binned_apl_overlay_plots(master_df)

    log_section("ANALYSIS COMPLETE")
    log_kv("Master CSV", MASTER_CSV)
    log_kv("Comparison CSV", COMPARISON_CSV)
    log_kv("Failed systems CSV", FAILED_CSV)
    log_kv("Output root", OUTPUT_ROOT)


if __name__ == "__main__":
    main()