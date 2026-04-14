#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APL analysis for PFPeptoids systems.

Creates:
1) masterdataanalysis.csv
2) bilayercomparisonanalysis.csv
3) failed_systems.csv
4) analysis_metadata.json
5) Per-system CSV + PNG outputs grouped by molecule/bilayer/run

Author-facing notes:
- This script never overwrites completed systems already present in masterdataanalysis.csv
- It analyzes only new valid systems
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
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

# =============================================================================
# USER CONFIGURATION
# =============================================================================
# 
# OPTIONAL MANUAL INPUT OVERRIDE
# Default = False
# False -> normal full scan from DATA_ROOT
# True  -> analyze only one target you provide
#
# Option A:
#   MANUAL_SYSTEM_DIR = folder containing Pull.gro Pull.xtc Relax.gro Relax.xtc
#
# Option B:
#   provide all four files directly
#
# If using direct files outside the normal folder structure, also set:
#   MANUAL_MOLECULE_NAME
#   MANUAL_SYSTEM_TYPE
#   MANUAL_RUN_NUMBER
# =============================================================================
MANUAL_MODE = False
MANUAL_SYSTEM_DIR = None

MANUAL_PULL_GRO = None
MANUAL_PULL_XTC = None
MANUAL_RELAX_GRO = None
MANUAL_RELAX_XTC = None

MANUAL_MOLECULE_NAME = "manual_system_1"
MANUAL_SYSTEM_TYPE = "manual_bilayer"
MANUAL_RUN_NUMBER = 0

SCRIPT_PATH = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/aplanalysis.py")
ANALYSIS_DIR = SCRIPT_PATH.parent
DATA_ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1")

MASTER_CSV = ANALYSIS_DIR / "masterdataanalysis.csv"
COMPARISON_CSV = ANALYSIS_DIR / "bilayercomparisonanalysis.csv"
FAILED_CSV = ANALYSIS_DIR / "failed_systems.csv"
METADATA_JSON = ANALYSIS_DIR / "analysis_metadata.json"
OUTPUT_ROOT = ANALYSIS_DIR / "output"

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

# Per-molecule bilayer comparison order
BILAYER_PRIORITY = [
    "POPC60CHOL40_W_WF",
    "POPC80POPS20_W_WF",
    "POPE75POPG25_W_WF",
]

# Plot styling — all user-customizable
PLOT_CFG = {
    "style_use_default_matplotlib": True,
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
    "stacked_combined_title": "Combined APL Comparison",

    "x_label": "Time (ns)",
    "y_label_po4": "APL", #note this is the PO4 method
    "y_label_box": "APL (box method)",

    "combined_use_box_method": True,   # False = PO4, True = BOX
    "single_use_box_method": True,     # False = PO4, True = BOX

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

def resolve_analysis_targets() -> List[dict]:
    """
    Returns a list of targets.
    Auto mode -> many systems from DATA_ROOT
    Manual mode -> one system from folder or direct files
    """
    if not MANUAL_MODE:
        return [{"system_dir": p, "manual": False} for p in scan_system_folders(DATA_ROOT)]

    if MANUAL_SYSTEM_DIR is not None:
        system_dir = Path(MANUAL_SYSTEM_DIR)
        if not system_dir.is_dir():
            raise NotADirectoryError(f"MANUAL_SYSTEM_DIR is not a directory: {system_dir}")
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
            "run_number": int(MANUAL_RUN_NUMBER),
        }]

    raise ValueError(
        "MANUAL_MODE=True but no valid manual input was provided. "
        "Use MANUAL_SYSTEM_DIR or all four direct files."
    )


def get_manual_system_output_dir(molecule_name: str, system_type: str, run_number: int) -> Path:
    return (
        OUTPUT_ROOT
        / sanitize_filename(molecule_name)
        / sanitize_filename(system_type)
        / f"run_{int(run_number)}"
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


def parse_run_number(run_folder_name: str) -> int:
    if not run_folder_name.startswith("run_"):
        raise ValueError(f"Run folder does not start with 'run_': {run_folder_name}")
    return int(run_folder_name.split("_", 1)[1])


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

    # Count phospholipid residues from PO4-bearing residues
    phospholipid_residues = po4.residues
    n_phospholipids = phospholipid_residues.n_residues

    # Count cholesterol residues directly
    chol_atoms = universe.select_atoms("resname " + " ".join(sorted(CHOL_RESNAMES)))
    n_chol = chol_atoms.residues.n_residues

    # Total membrane molecules
    n_membrane_molecules = n_phospholipids + n_chol

    # Fallback for systems where CHOL exists conceptually but is not detected
    if "CHOL" in system_type and n_chol == 0:
        assumed_fraction = SYSTEM_TYPE_TO_ASSUMED_PHOSPHOLIPID_FRACTION.get(system_type)
        if assumed_fraction is None or assumed_fraction <= 0 or assumed_fraction > 1:
            raise ValueError(
                f"Could not detect CHOL residues and no valid fallback fraction defined for {system_type}"
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

        # Wrap PO4 coordinates into the primary orthorhombic box for continuity with the existing method
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
    ax.set_title(title)
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

    combined_metric = "APL_BOX" if PLOT_CFG["combined_use_box_method"] else "APL_PO4"

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

    ax.set_title(PLOT_CFG["combined_title"])
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


def stitch_images_horizontally(image_paths: List[Path], output_path: Path, background=(255, 255, 255)) -> None:
    valid_paths = [p for p in image_paths if p.exists()]
    if not valid_paths:
        return

    images = [plt.imread(str(p)) for p in valid_paths]

    # convert to PIL-compatible uint8 arrays if needed
    pil_images = []
    from PIL import Image
    import numpy as np

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
    """
    Generates only the requested ordered bilayer differences.
    Left minus right order follows user specification.
    """
    pairs = []
    present = [b for b in BILAYER_PRIORITY if b in existing_bilayers]

    if "POPC60CHOL40_W_WF" in present and "POPC80POPS20_W_WF" in present:
        pairs.append(("POPC80POPS20_W_WF", "POPC60CHOL40_W_WF"))
    if "POPC60CHOL40_W_WF" in present and "POPE75POPG25_W_WF" in present:
        pairs.append(("POPE75POPG25_W_WF", "POPC60CHOL40_W_WF"))
    if "POPC80POPS20_W_WF" in present and "POPE75POPG25_W_WF" in present:
        pairs.append(("POPE75POPG25_W_WF", "POPC80POPS20_W_WF"))

    return pairs


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

        # Comparisons within same molecule, same run_number, same nmol
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
                    "run_number": int(run_number),
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
    For each molecule + run + nmol, stitch combined APL graphs from each bilayer horizontally.
    """
    if master_df.empty:
        return

    for (molecule_name, run_number, nmol), sub in master_df.groupby(["molecule_name", "run_number", "nmol"], dropna=False):
        bilayers = ordered_bilayers(list(sub["system_type"].unique()))
        image_paths = []

        for bilayer in bilayers:
            row = sub[sub["system_type"] == bilayer]
            if row.empty:
                continue

            system_output_dir = (
                OUTPUT_ROOT
                / sanitize_filename(molecule_name)
                / sanitize_filename(bilayer)
                / f"run_{int(run_number)}"
            )

            combined_png = system_output_dir / "combined_APL_over_time.png"
            if combined_png.exists():
                image_paths.append(combined_png)

        if not image_paths:
            continue

        stacked_dir = OUTPUT_ROOT / sanitize_filename(molecule_name)
        stacked_name = f"stacked_combined_APL_run_{int(run_number)}_nmol_{int(nmol)}.png"
        stacked_path = stacked_dir / stacked_name

        # proper check: don't recreate if already exists
        if stacked_path.exists():
            continue

        stitch_images_horizontally(image_paths, stacked_path)


def scan_system_folders(root: Path) -> List[Path]:
    folders = []
    for bilayer_dir in sorted(root.iterdir()):
        if not bilayer_dir.is_dir():
            continue

        for run_dir in sorted(bilayer_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            for system_dir in sorted(run_dir.iterdir()):
                if system_dir.is_dir():
                    folders.append(system_dir)

    return folders


def get_system_output_dir(molecule_name: str, system_type: str, run_number: int) -> Path:
    return (
        OUTPUT_ROOT
        / sanitize_filename(molecule_name)
        / sanitize_filename(system_type)
        / f"run_{run_number}"
    )


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
        run_number = int(manual_target["run_number"])
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

    single_metric = "APL_BOX" if PLOT_CFG["single_use_box_method"] else "APL_PO4"

    if not image_outputs_exist(system_output_dir):
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

    row = {
        "system_key": make_system_key(molecule_name, nmol, system_type, run_number),

        "molecule_name": molecule_name,
        "molecule_type": molecule_type,
        "nmol": nmol,
        "length": length,
        "molecular_weight": mw,
        "molecular_weight_missing_codes": ";".join(missing_mass_codes),

        "positive_charge": pos_charge,
        "negative_charge": neg_charge,
        "net_charge": net_charge,
        "charge_density": charge_density,

        "system_type": system_type,
        "run_number": run_number,

        "start_APL_PO4": start_apl_po4,
        "finish_APL_PO4_100ns": finish_apl_po4,
        "APL_change_PO4": start_apl_po4 - finish_apl_po4,
        "APL_percent_change_PO4": ((finish_apl_po4 - start_apl_po4) / start_apl_po4 * 100.0) if start_apl_po4 != 0 else np.nan,
        "APL_mean_last10ns_PO4": apl_po4_mean_last10,
        "APL_std_last10ns_PO4": apl_po4_std_last10,

        "start_APL_BOX": start_apl_box,
        "finish_APL_BOX_100ns": finish_apl_box,
        "APL_change_BOX": start_apl_box - finish_apl_box,
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

        "has_missing_mass": len(missing_mass_codes) > 0,
        "missing_mass_count": len(missing_mass_codes),
        "relax_end_time_ns": relax_end_ns,
        "reached_100ns": abs(nearest_time_ns - TARGET_RELAX_TIME_NS) <= TARGET_RELAX_TOLERANCE_NS,
        "target_relax_time_ns": TARGET_RELAX_TIME_NS,
        "remaining_relax_time_ns": max(0.0, TARGET_RELAX_TIME_NS - relax_end_ns),
        "token_sequence": "-".join(tokens) if molecule_type == "peptoid" else "".join(tokens),

        "component_count": len(parse_system_components(molecule_name)),
        "component_names": "|".join(str(c["name"]) for c in parse_system_components(molecule_name)),
        "component_counts": "|".join(str(int(c["count"])) for c in parse_system_components(molecule_name)),
        "contains_mixture": len(parse_system_components(molecule_name)) > 1,

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
        "master_csv": str(MASTER_CSV),
        "comparison_csv": str(COMPARISON_CSV),
        "failed_csv": str(FAILED_CSV),
        "output_root": str(OUTPUT_ROOT),
        "target_relax_time_ns": TARGET_RELAX_TIME_NS,
        "target_relax_tolerance_ns": TARGET_RELAX_TOLERANCE_NS,
        "po4_selection": PO4_SELECTION,
        "bilayer_priority": BILAYER_PRIORITY,
        "plot_config": PLOT_CFG,
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

    existing_master = safe_read_csv(MASTER_CSV)
    existing_failed = safe_read_csv(FAILED_CSV)

    existing_keys = set()
    if not existing_master.empty and "system_key" in existing_master.columns:
        existing_keys = set(existing_master["system_key"].astype(str).tolist())

    new_rows = []
    failed_records = []

    targets = resolve_analysis_targets()

    for target in targets:
        system_dir = target.get("system_dir", None)
        manual_target = target if target.get("manual", False) else None

        try:
            if manual_target is None:
                system_type = system_dir.parents[1].name
                run_number = parse_run_number(system_dir.parent.name)
                molecule_name, nmol = split_system_folder_name(system_dir.name)
            else:
                system_type = manual_target["system_type"]
                run_number = int(manual_target["run_number"])
                molecule_name = str(manual_target["molecule_name"])
                _, nmol = split_system_folder_name(molecule_name)

            system_key = make_system_key(molecule_name, nmol, system_type, run_number)
            system_output_dir = get_system_output_dir(molecule_name, system_type, run_number)

            row_missing = system_key not in existing_keys
            plots_missing = not image_outputs_exist(system_output_dir)

            if not row_missing and not plots_missing:
                continue

            row = analyze_system(
                system_dir=system_dir,
                peptide_mass=peptide_mass,
                peptoid_mass=peptoid_mass,
                manual_target=manual_target,
            )

            if row_missing:
                new_rows.append(row)

        except Exception as exc:
            try:
                if manual_target is None:
                    system_type = system_dir.parents[1].name
                    run_number = parse_run_number(system_dir.parent.name)
                    molecule_name, nmol = split_system_folder_name(system_dir.name)
                    failure_path = system_dir
                else:
                    system_type = manual_target["system_type"]
                    run_number = int(manual_target["run_number"])
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

    print("Analysis complete.")
    print(f"Master CSV: {MASTER_CSV}")
    print(f"Comparison CSV: {COMPARISON_CSV}")
    print(f"Failed systems CSV: {FAILED_CSV}")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()