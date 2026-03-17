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

# Use PO4 beads for PO4-style APL and thickness
PO4_SELECTION = "name PO4"

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
    "x_label": "Time (ns)",
    "y_label_po4": "APL (PO4 method)",
    "y_label_box": "APL (box method)",
    "combined_use_box_method": False,  # False = plot PO4 APL, True = plot BOX APL
    "pull_xlim": None,
    "relax_xlim": None,
    "combined_xlim": None,
    "pull_ylim": None,
    "relax_ylim": None,
    "combined_ylim": None,
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


def detect_molecule_type(molecule_name: str) -> str:
    return "peptoid" if "-" in molecule_name else "peptide"


def split_system_folder_name(folder_name: str) -> Tuple[str, int]:
    """
    Example:
        RWWRWW_80 -> ("RWWRWW", 80)
        Nr-Nw-Nw-Nr-Nw-Nw_80 -> ("Nr-Nw-Nw-Nr-Nw-Nw", 80)
    """
    if "_" not in folder_name:
        raise ValueError(f"Cannot parse molecule/nmol from folder name: {folder_name}")

    molecule_name, nmol_str = folder_name.rsplit("_", 1)
    nmol = int(nmol_str)
    return molecule_name, nmol


def parse_run_number(run_folder_name: str) -> int:
    if not run_folder_name.startswith("run_"):
        raise ValueError(f"Run folder does not start with 'run_': {run_folder_name}")
    return int(run_folder_name.split("_", 1)[1])


def tokenize_molecule(molecule_name: str, molecule_type: str) -> List[str]:
    if molecule_type == "peptoid":
        return [x.strip() for x in molecule_name.split("-") if x.strip()]
    return list(molecule_name.strip())


def charge_counts(tokens: List[str], molecule_type: str) -> Tuple[int, int, int]:
    if molecule_type == "peptide":
        pos = sum(tok in {"K", "R", "H"} for tok in tokens)
        neg = sum(tok in {"D", "E"} for tok in tokens)
    else:
        pos = sum(tok in {"Nk", "Nr", "Nke"} for tok in tokens)
        neg = sum(tok in {"Nd", "Ne"} for tok in tokens)

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


def make_system_key(molecule_name: str, nmol: int, system_type: str, run_number: int) -> str:
    return f"{molecule_name}|{nmol}|{system_type}|{run_number}"


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
) -> None:
    failed_records.append({
        "molecule_name": molecule_name,
        "nmol": nmol,
        "system_type": system_type,
        "run_number": run_number,
        "stage": stage,
        "reason": reason,
        "system_path": "" if system_path is None else str(system_path),
    })


def get_po4_and_leaflets(universe: mda.Universe):
    po4 = universe.select_atoms(PO4_SELECTION)
    n_lipids = len(po4)

    if n_lipids == 0:
        raise ValueError("No PO4 atoms found.")

    lipids_per_leaflet = n_lipids / 2.0
    return po4, n_lipids, lipids_per_leaflet


def analyze_trajectory(
    gro_path: Path,
    xtc_path: Path,
    phase_name: str,
) -> pd.DataFrame:
    """
    Returns one DataFrame with all needed time-series data for this phase.
    Opens the trajectory once only.
    """
    u = mda.Universe(str(gro_path), str(xtc_path), in_memory=False)
    po4, n_lipids, lipids_per_leaflet = get_po4_and_leaflets(u)

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
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_CFG["figure_size"], dpi=PLOT_CFG["dpi"])
    ax.plot(df["time_ns"], df[y_col], linewidth=PLOT_CFG["line_width"])
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
        label="Pull",
    )
    ax.plot(
        relax_plot["time_ns_combined"],
        relax_plot[combined_metric],
        linewidth=PLOT_CFG["line_width"],
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
    out.to_csv(output_csv, index=False)


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
        mol_df_sorted.to_csv(molecule_dir / "master_rows_for_this_molecule.csv", index=False)

        # Comparisons within same molecule, same run_number, same nmol
        for (run_number, nmol), sub in mol_df.groupby(["run_number", "nmol"], dropna=False):
            bilayers = list(sub["system_type"].unique())
            for left_bilayer, right_bilayer in ordered_bilayer_pairs(bilayers):
                left = sub[sub["system_type"] == left_bilayer]
                right = sub[sub["system_type"] == right_bilayer]

                if left.empty or right.empty:
                    continue

                left_row = left.iloc[0]
                right_row = right.iloc[0]

                comparison_rows.append({
                    "molecule_name": molecule_name,
                    "molecule_type": left_row["molecule_type"],
                    "run_number": int(run_number),
                    "nmol": int(nmol),

                    "bilayer_left": left_bilayer,
                    "bilayer_right": right_bilayer,

                    "finish_APL_PO4_left": left_row["finish_APL_PO4_100ns"],
                    "finish_APL_PO4_right": right_row["finish_APL_PO4_100ns"],
                    "finish_APL_PO4_difference_left_minus_right":
                        left_row["finish_APL_PO4_100ns"] - right_row["finish_APL_PO4_100ns"],

                    "finish_APL_BOX_left": left_row["finish_APL_BOX_100ns"],
                    "finish_APL_BOX_right": right_row["finish_APL_BOX_100ns"],
                    "finish_APL_BOX_difference_left_minus_right":
                        left_row["finish_APL_BOX_100ns"] - right_row["finish_APL_BOX_100ns"],

                    "start_APL_PO4_left": left_row["start_APL_PO4"],
                    "start_APL_PO4_right": right_row["start_APL_PO4"],
                    "finish_diff_vs_start_left_PO4_percent":
                        ((left_row["finish_APL_PO4_100ns"] - right_row["finish_APL_PO4_100ns"]) / left_row["start_APL_PO4"] * 100.0)
                        if pd.notna(left_row["start_APL_PO4"]) and left_row["start_APL_PO4"] != 0 else np.nan,
                    "finish_diff_vs_start_right_PO4_percent":
                        ((left_row["finish_APL_PO4_100ns"] - right_row["finish_APL_PO4_100ns"]) / right_row["start_APL_PO4"] * 100.0)
                        if pd.notna(right_row["start_APL_PO4"]) and right_row["start_APL_PO4"] != 0 else np.nan,

                    "start_APL_BOX_left": left_row["start_APL_BOX"],
                    "start_APL_BOX_right": right_row["start_APL_BOX"],
                    "finish_diff_vs_start_left_BOX_percent":
                        ((left_row["finish_APL_BOX_100ns"] - right_row["finish_APL_BOX_100ns"]) / left_row["start_APL_BOX"] * 100.0)
                        if pd.notna(left_row["start_APL_BOX"]) and left_row["start_APL_BOX"] != 0 else np.nan,
                    "finish_diff_vs_start_right_BOX_percent":
                        ((left_row["finish_APL_BOX_100ns"] - right_row["finish_APL_BOX_100ns"]) / right_row["start_APL_BOX"] * 100.0)
                        if pd.notna(right_row["start_APL_BOX"]) and right_row["start_APL_BOX"] != 0 else np.nan,
                })

    comparison_df = pd.DataFrame(comparison_rows)

    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(
            ["molecule_name", "run_number", "nmol", "bilayer_left", "bilayer_right"]
        ).reset_index(drop=True)

        for molecule_name, sub in comparison_df.groupby("molecule_name", sort=True):
            molecule_dir = OUTPUT_ROOT / sanitize_filename(molecule_name)
            molecule_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(molecule_dir / "bilayercomparison_for_this_molecule.csv", index=False)

    return comparison_df


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


def analyze_system(
    system_dir: Path,
    peptide_mass: Dict[str, float],
    peptoid_mass: Dict[str, float],
) -> dict:
    """
    Analyze one completed system and return one master row.
    Raises exceptions for failures that should be logged into failed_systems.csv.
    """
    system_type = system_dir.parents[1].name
    run_number = parse_run_number(system_dir.parent.name)
    molecule_name, nmol = split_system_folder_name(system_dir.name)
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

    pull_gro = system_dir / "Pull.gro"
    pull_xtc = system_dir / "Pull.xtc"
    relax_gro = system_dir / "Relax.gro"
    relax_xtc = system_dir / "Relax.xtc"

    required = [pull_gro, pull_xtc, relax_gro, relax_xtc]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {missing}")

    pull_df = analyze_trajectory(pull_gro, pull_xtc, phase_name="pull")
    relax_df = analyze_trajectory(relax_gro, relax_xtc, phase_name="relax")

    # Validate 100 ns availability
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

    # Save grouped per-system outputs
    system_output_dir = (
        OUTPUT_ROOT
        / sanitize_filename(molecule_name)
        / sanitize_filename(system_type)
        / f"run_{run_number}"
    )
    system_output_dir.mkdir(parents=True, exist_ok=True)

    save_system_timeseries(
        pull_df=pull_df,
        relax_df=relax_df,
        output_csv=system_output_dir / "APL_time_series.csv",
    )

    plot_single_phase(
        df=pull_df,
        output_png=system_output_dir / "pull_APL_vs_time.png",
        title=PLOT_CFG["pull_title"],
        y_col="APL_PO4",
        xlim=PLOT_CFG["pull_xlim"],
        ylim=PLOT_CFG["pull_ylim"],
    )

    plot_single_phase(
        df=relax_df,
        output_png=system_output_dir / "relax_APL_vs_time.png",
        title=PLOT_CFG["relax_title"],
        y_col="APL_PO4",
        xlim=PLOT_CFG["relax_xlim"],
        ylim=PLOT_CFG["relax_ylim"],
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

        "system_dir": str(system_dir),
        "output_dir": str(system_output_dir),
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

    system_dirs = scan_system_folders(DATA_ROOT)

    for system_dir in system_dirs:
        try:
            system_type = system_dir.parents[1].name
            run_number = parse_run_number(system_dir.parent.name)
            molecule_name, nmol = split_system_folder_name(system_dir.name)
            system_key = make_system_key(molecule_name, nmol, system_type, run_number)

            # Skip if already analyzed successfully
            if system_key in existing_keys:
                continue

            row = analyze_system(
                system_dir=system_dir,
                peptide_mass=peptide_mass,
                peptoid_mass=peptoid_mass,
            )
            new_rows.append(row)

        except Exception as exc:
            # Try to preserve as much identity as possible even on failure
            try:
                system_type = system_dir.parents[1].name
            except Exception:
                system_type = None

            try:
                run_number = parse_run_number(system_dir.parent.name)
            except Exception:
                run_number = None

            try:
                molecule_name, nmol = split_system_folder_name(system_dir.name)
            except Exception:
                molecule_name, nmol = system_dir.name, None

            append_failed(
                failed_records=failed_records,
                molecule_name=molecule_name,
                nmol=nmol,
                system_type=system_type,
                run_number=run_number,
                stage="analysis",
                reason=f"{type(exc).__name__}: {exc}",
                system_path=system_dir,
            )

    # Update master CSV
    if new_rows:
        new_master = pd.DataFrame(new_rows)
        master_df = pd.concat([existing_master, new_master], ignore_index=True)
    else:
        master_df = existing_master.copy()

    if not master_df.empty:
        master_df = master_df.drop_duplicates(subset=["system_key"], keep="first")
        master_df = master_df.sort_values(["molecule_name", "system_type", "run_number", "nmol"]).reset_index(drop=True)
    safe_write_csv(master_df, MASTER_CSV)

    # Update failed CSV
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

    # Rebuild grouped outputs + bilayer comparisons from full master CSV
    comparison_df = build_grouped_outputs(master_df)
    safe_write_csv(comparison_df, COMPARISON_CSV)

    print(f"Analysis complete.")
    print(f"Master CSV: {MASTER_CSV}")
    print(f"Comparison CSV: {COMPARISON_CSV}")
    print(f"Failed systems CSV: {FAILED_CSV}")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()