#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SASA_NVT_to_CpHMD_delta.py

Purpose
-------
Compare SASA/exposure between:

    initial state = first frame of a GROMACS NVT trajectory
    final state   = last frame of a NAMD CpHMD trajectory

This produces one clean delta-SASA table:

    final_SASA - initial_SASA

Designed for PFPeptoids / MARTINI-style coarse-grained systems.
For coarse-grained systems, treat the output as a consistent exposure metric,
not a strict atomistic SASA value.

Main output
-----------
    sasa_nvt_first_to_cphmd_last_delta.csv

Useful secondary outputs
------------------------
    sasa_single_frame_measurements.csv
    sasa_failures.csv
    sasa_settings.json
"""

from pathlib import Path
import csv
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import MDAnalysis as mda
from scipy.spatial import cKDTree


# =============================================================================
# USER SETTINGS
# =============================================================================

# Put either:
#   1) one exact GROMACS system folder, or
#   2) a higher root containing many systems.
#
# The script only uses files with "nvt" in the .xtc filename.
GROMACS_ROOT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1/POPC60CHOL40_W_WF/run_0/FLPLLASLFSRLF_80"
)

# Put either:
#   1) one exact NAMD/CpHMD system folder, or
#   2) a root containing many CpHMD folders.
#
# The script uses .dcd files and takes the final frame.
NAMD_ROOT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/Full_PRC_Setup/Working_Directory/CpHMD_Runs"
)

OUTPUT_DIR = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/results/SASA"
)

# If None, search all subfolders.
# If an integer, restrict recursive search depth.
MAX_DEPTH = None

# For the current scientific question, these are fixed:
#   initial = first frame of GROMACS NVT
#   final   = last frame of NAMD CpHMD
# Initial GROMACS structure to use.
# Your current folder has Box.gro, Min.gro, Pull.gro, Relax.gro.
# For the most untouched box, use Box.gro.

GROMACS_INITIAL_STRUCTURE = "Box.gro"
GROMACS_FRAME_INDEX = 0
NAMD_FRAME_INDEX = -1


# =============================================================================
# TARGET SELECTIONS
# =============================================================================

# TARGET = what surface you measure.
# Example:
#   target = polymer_all -> measure peptide/peptoid exposed surface.
#   target = bilayer     -> measure bilayer exposed surface.
TARGET_SELECTIONS = {
    "polymer_all": (
        "resname "
        "ALAC ALAN ALA VALC VALN VAL ILEC ILEN ILE LEUC LEUN LEU "
        "METC METN MET CYSC CYSN CYS PROC PRON PRO SERC SERN SER "
        "THRC THRN THR ASNC ASNN ASN GLNC GLNN GLN HSEC HSEN HSE "
        "ASPC ASPN ASP GLUC GLUN GLU PHEC PHEN PHE TYRC TYRN TYR "
        "TRPC TRPN TRP LYSC LYSN LYS ARGC ARGN ARG "
        "Na Nab Ni Nl Nm Nv Nn Nq Ns Nse Nt NmO Nd Ne "
        "Nf Nfn Nfe Nfex Nw Nwe Ny Nfe4Cl Nfe4Br Nfnaph Nk Nke Nr"
    ),

    "peptide": (
        "resname "
        "ALAC ALAN ALA VALC VALN VAL ILEC ILEN ILE LEUC LEUN LEU "
        "METC METN MET CYSC CYSN CYS PROC PRON PRO SERC SERN SER "
        "THRC THRN THR ASNC ASNN ASN GLNC GLNN GLN HSEC HSEN HSE "
        "ASPC ASPN ASP GLUC GLUN GLU PHEC PHEN PHE TYRC TYRN TYR "
        "TRPC TRPN TRP LYSC LYSN LYS ARGC ARGN ARG"
    ),

    "peptoid": (
        "resname "
        "Na Nab Ni Nl Nm Nv Nn Nq Ns Nse Nt NmO Nd Ne "
        "Nf Nfn Nfe Nfex Nw Nwe Ny Nfe4Cl Nfe4Br Nfnaph Nk Nke Nr"
    ),

    "bilayer": "resname POPC POPE POPG POPS CHOL DPPC DTPC",

    "POPC": "resname POPC",
    "POPE": "resname POPE",
    "POPG": "resname POPG",
    "POPS": "resname POPS",
    "CHOL": "resname CHOL",
}

# Targets to calculate.
TARGETS_TO_RUN = [
    "polymer_all",
    "bilayer",
]

# OCCLUDER/BLOCKER = what can hide the target surface.
#
# Recommended default for your current analysis:
#   exclude explicit water and ions from blocking the surface.
#
# Interpretation:
#   polymer_all SASA = polymer surface exposed to non-polymer/non-water environment
#   bilayer SASA     = bilayer surface exposed to non-water environment
#
# If this selection fails due to unusual residue names, change it to:
#   OCCLUDER_SELECTION = "all"
OCCLUDER_SELECTION = "not resname W WF SW TW H2O NA CL ION"


# =============================================================================
# SASA / EXPOSURE SETTINGS
# =============================================================================

# Probe radius in Angstrom.
# 1.4 Å is the conventional atomistic water probe.
# For MARTINI this is still useful as a consistent metric, not an absolute truth.
PROBE_RADIUS = 1.4

# More points = more accurate but slower.
# Try 48 for fast testing, 96 for normal use, 192 for slower validation.
N_SPHERE_POINTS = 96

# MARTINI / coarse-grained default bead radius in Angstrom.
DEFAULT_RADIUS = 2.35

# Smaller radius for solvent/ions if they are ever included in target/occluder selections.
SMALL_RADIUS = 1.9
SMALL_RESNAMES = {"W", "WF", "SW", "TW", "H2O", "NA", "CL", "ION"}


# =============================================================================
# FILE DISCOVERY HELPERS
# =============================================================================

def path_depth(path: Path, root: Path) -> int:
    """Return path depth relative to root. Used only if MAX_DEPTH is set."""
    try:
        return len(path.relative_to(root).parts)
    except Exception:
        return 999999


def allowed_depth(path: Path, root: Path) -> bool:
    """Return True if path is inside the allowed recursive search depth."""
    if MAX_DEPTH is None:
        return True
    return path_depth(path, root) <= MAX_DEPTH


def natural_sort_key(path: Path):
    """Human-friendly sorting for filenames containing numbers."""
    text = str(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def infer_bilayer_from_path(folder: Path) -> str:
    """
    Infer bilayer from a GROMACS-style path:
        .../<bilayer>/run_0/<system>
    """
    try:
        if folder.parent.name.startswith("run_"):
            return folder.parents[1].name
    except Exception:
        pass
    return "unknown_bilayer"


def infer_run_from_path(folder: Path) -> str:
    """Infer run name from a GROMACS-style path."""
    try:
        if folder.parent.name.startswith("run_"):
            return folder.parent.name
    except Exception:
        pass
    return "unknown_run"


def find_first_existing(candidates: List[Path]) -> Optional[Path]:
    """Return first existing path from a list, otherwise None."""
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


# =============================================================================
# GROMACS DISCOVERY
# =============================================================================

def find_gromacs_initial_systems(root: Path) -> List[Dict[str, object]]:
    """
    Find GROMACS initial structures.

    For your current question:
        initial = Box.gro

    This is a static single-frame structure, so there is no trajectory file.
    """

    systems: List[Dict[str, object]] = []

    if not root.exists():
        print(f"WARNING: GROMACS root does not exist: {root}")
        return systems

    candidate_folders = set()

    # Case 1: root itself is one system folder
    if (root / GROMACS_INITIAL_STRUCTURE).exists():
        candidate_folders.add(root)

    # Case 2: root contains many system folders
    for p in root.rglob(GROMACS_INITIAL_STRUCTURE):
        if not allowed_depth(p, root):
            continue
        candidate_folders.add(p.parent)

    for folder in sorted(candidate_folders, key=natural_sort_key):
        topology = folder / GROMACS_INITIAL_STRUCTURE

        if not topology.exists():
            continue

        systems.append({
            "engine": "GROMACS",
            "system_name": folder.name,
            "bilayer": infer_bilayer_from_path(folder),
            "run_name": infer_run_from_path(folder),
            "folder": folder,
            "topology": topology,
            "trajectory": None,
            "topology_file": topology.name,
            "trajectory_file": "",
            "trajectory_type": "static_gro",
            "stage": "GROMACS_Box_gro_first",
            "frame_index": 0,
        })

    return systems


# =============================================================================
# NAMD / CPHMD DISCOVERY
# =============================================================================

def find_namd_topology_for_folder(folder: Path) -> Optional[Path]:
    """
    Find a topology for a NAMD DCD folder.

    Prefer PSF. Fall back to PDB.
    Search current folder first, then parent and grandparent.
    This handles layouts like:
        CpHMD_system/pH7.4/*.dcd
        CpHMD_system/*.psf
    """
    search_folders = [folder, folder.parent, folder.parent.parent]

    for search_folder in search_folders:
        if not search_folder.exists():
            continue
        psf = sorted(search_folder.glob("*.psf"), key=natural_sort_key)
        if psf:
            return psf[0]

    for search_folder in search_folders:
        if not search_folder.exists():
            continue
        pdb = sorted(search_folder.glob("*.pdb"), key=natural_sort_key)
        if pdb:
            return pdb[0]

    return None


def find_namd_cphmd_systems(root: Path) -> List[Dict[str, object]]:
    """
    Find NAMD/CpHMD systems.

    A valid NAMD final-state system needs:
        .dcd trajectory
        .psf topology, or .pdb fallback

    If both parent and child folders contain DCD files, keep the deepest folder
    to avoid duplicate parent/child analysis.
    """
    systems: List[Dict[str, object]] = []

    if not root.exists():
        print(f"WARNING: NAMD root does not exist: {root}")
        return systems

    candidate_folders = set()

    for dcd in root.rglob("*.dcd"):
        if not allowed_depth(dcd, root):
            continue
        candidate_folders.add(dcd.parent)

    filtered_folders = []
    for folder in sorted(candidate_folders, key=natural_sort_key):
        is_parent_of_another = any(
            other != folder and str(other).startswith(str(folder) + "/")
            for other in candidate_folders
        )
        if not is_parent_of_another:
            filtered_folders.append(folder)

    for folder in filtered_folders:
        dcd_candidates = sorted(folder.glob("*.dcd"), key=natural_sort_key)
        if not dcd_candidates:
            continue

        trajectory = dcd_candidates[0]
        topology = find_namd_topology_for_folder(folder)

        if topology is None:
            continue

        # If folder is pH7.4, use parent folder as the real system name.
        if folder.name.lower().startswith("ph"):
            system_name = folder.parent.name
        else:
            system_name = folder.name

        systems.append({
            "engine": "NAMD",
            "system_name": system_name,
            "bilayer": "unknown_bilayer",
            "run_name": "unknown_run",
            "folder": folder,
            "topology": topology,
            "trajectory": trajectory,
            "topology_file": topology.name,
            "trajectory_file": trajectory.name,
            "trajectory_type": "CpHMD",
            "stage": "NAMD_CpHMD_last",
            "frame_index": NAMD_FRAME_INDEX,
        })

    return systems


# =============================================================================
# SASA CALCULATION
# =============================================================================

def sphere_points_fibonacci(n_points: int) -> np.ndarray:
    """Generate deterministic approximately-even points on a unit sphere."""
    if n_points < 12:
        raise ValueError("N_SPHERE_POINTS should be at least 12.")

    points = []
    offset = 2.0 / n_points
    increment = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n_points):
        y = ((i * offset) - 1.0) + offset / 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append((x, y, z))

    return np.asarray(points, dtype=float)


SPHERE_POINTS = sphere_points_fibonacci(N_SPHERE_POINTS)


def radius_array(atomgroup) -> np.ndarray:
    """
    Assign CG radii.

    This is a consistent MARTINI-style exposure metric:
        most beads: DEFAULT_RADIUS
        water/ions: SMALL_RADIUS
    """
    radii = np.full(atomgroup.n_atoms, DEFAULT_RADIUS, dtype=float)

    for i, resname in enumerate(atomgroup.resnames):
        if str(resname).strip() in SMALL_RESNAMES:
            radii[i] = SMALL_RADIUS

    return radii


def calculate_target_sasa(target_ag, occluder_ag) -> float:
    """
    Shrake-Rupley SASA/exposure calculation.

    target_ag:
        atoms/beads whose surface is being measured.

    occluder_ag:
        atoms/beads that can hide/block the target surface.

    In the recommended default, water and ions are excluded from occluders.
    """
    if target_ag.n_atoms == 0:
        return np.nan

    if occluder_ag.n_atoms == 0:
        raise ValueError("Occluder selection has 0 atoms.")

    target_coords = np.asarray(target_ag.positions, dtype=float)
    occluder_coords = np.asarray(occluder_ag.positions, dtype=float)

    target_indices = np.asarray(target_ag.indices, dtype=int)
    occluder_indices = np.asarray(occluder_ag.indices, dtype=int)

    target_radii = radius_array(target_ag) + PROBE_RADIUS
    occluder_radii = radius_array(occluder_ag) + PROBE_RADIUS

    max_occluder_radius = float(np.max(occluder_radii))
    tree = cKDTree(occluder_coords)

    total_area = 0.0

    for i in range(target_ag.n_atoms):
        center = target_coords[i]
        atom_index = target_indices[i]
        ri = target_radii[i]

        surface_points = center + SPHERE_POINTS * ri
        candidate_ids = tree.query_ball_point(center, r=ri + max_occluder_radius)

        if not candidate_ids:
            accessible_count = N_SPHERE_POINTS
        else:
            accessible = np.ones(N_SPHERE_POINTS, dtype=bool)

            for cid in candidate_ids:
                # Do not let the atom/bead hide its own surface.
                if occluder_indices[cid] == atom_index:
                    continue

                occluder_center = occluder_coords[cid]
                occluder_radius = occluder_radii[cid]

                diff = surface_points - occluder_center
                dist2 = np.einsum("ij,ij->i", diff, diff)

                accessible &= dist2 > occluder_radius * occluder_radius

                if not accessible.any():
                    break

            accessible_count = int(accessible.sum())

        atom_area = 4.0 * np.pi * ri * ri * accessible_count / float(N_SPHERE_POINTS)
        total_area += atom_area

    return float(total_area)


# =============================================================================
# SINGLE-FRAME ANALYSIS
# =============================================================================

def resolve_frame_index(requested_index: int, n_frames: int) -> int:
    """Convert negative frame indices into absolute frame indices."""
    if requested_index < 0:
        resolved = n_frames + requested_index
    else:
        resolved = requested_index

    if resolved < 0 or resolved >= n_frames:
        raise ValueError(
            f"Invalid frame index {requested_index}. Resolved frame = {resolved}; "
            f"trajectory has {n_frames} frames."
        )

    return int(resolved)


def calculate_single_frame_sasa(system: Dict[str, object], target_name: str) -> Dict[str, object]:
    """
    Calculate SASA for one system, one target, one frame.

    For this script:
        GROMACS -> first frame of NVT trajectory
        NAMD    -> last frame of CpHMD trajectory
    """
    topology = Path(system["topology"])
    trajectory = system["trajectory"]
    trajectory = Path(trajectory) if trajectory else None
    requested_frame = int(system["frame_index"])

    selection = TARGET_SELECTIONS[target_name]

    print(f"\nReading {system['engine']}: {system['folder']}")
    print(f"  stage:      {system['stage']}")
    print(f"  topology:   {topology.name}")
    print(f"  trajectory: {trajectory.name if trajectory is not None else 'static structure only'}")
    print(f"  target:     {target_name}")

    if trajectory is None:
        # Static structure only, e.g. Box.gro.
        u = mda.Universe(str(topology))
    else:
        u = mda.Universe(str(topology), str(trajectory), in_memory=False)
    
    n_frames = len(u.trajectory)
    resolved_frame = resolve_frame_index(requested_frame, n_frames)
    
    ts = u.trajectory[resolved_frame]

    target_ag = u.select_atoms(selection)
    occluder_ag = u.select_atoms(OCCLUDER_SELECTION)

    if target_ag.n_atoms == 0:
        raise ValueError(f"Target selection has 0 atoms: {target_name} = {selection}")

    if occluder_ag.n_atoms == 0:
        raise ValueError(f"Occluder selection has 0 atoms: {OCCLUDER_SELECTION}")

    sasa = calculate_target_sasa(target_ag, occluder_ag)

    return {
        "engine": system["engine"],
        "system_name": system["system_name"],
        "bilayer": system.get("bilayer", ""),
        "run_name": system.get("run_name", ""),
        "stage": system["stage"],
        "trajectory_type": system["trajectory_type"],

        "folder": str(system["folder"]),
        "topology_file": topology.name,
        "trajectory_file": trajectory.name if trajectory is not None else "",
        "topology_path": str(topology),
        "trajectory_path": str(trajectory) if trajectory is not None else "",

        "target": target_name,
        "target_selection": selection,
        "occluder_selection": OCCLUDER_SELECTION,

        "requested_frame_index": requested_frame,
        "frame_index": resolved_frame,
        "n_frames_total": int(n_frames),
        "time_ps": float(ts.time),
        "time_ns": float(ts.time) / 1000.0,

        "n_target_atoms": int(target_ag.n_atoms),
        "n_occluder_atoms": int(occluder_ag.n_atoms),

        "SASA": float(sasa),
        "SASA_per_atom": float(sasa) / int(target_ag.n_atoms),

        "probe_radius": float(PROBE_RADIUS),
        "default_radius": float(DEFAULT_RADIUS),
        "n_sphere_points": int(N_SPHERE_POINTS),
    }


# =============================================================================
# SYSTEM MATCHING
# =============================================================================

def match_score(gmx: Dict[str, object], namd: Dict[str, object]) -> int:
    """Score how likely a NAMD system matches a GROMACS system."""
    gmx_name = str(gmx["system_name"])
    gmx_bilayer = str(gmx.get("bilayer", ""))
    gmx_run = str(gmx.get("run_name", ""))
    namd_name = str(namd["system_name"])
    namd_folder = str(namd["folder"])
    namd_text = f"{namd_name} {namd_folder}"

    score = 0

    if gmx_name and gmx_name in namd_text:
        score += 100
    if gmx_bilayer and gmx_bilayer != "unknown_bilayer" and gmx_bilayer in namd_text:
        score += 50
    if gmx_run and gmx_run != "unknown_run" and gmx_run in namd_text:
        score += 20

    return score


def match_gromacs_to_namd(gromacs_system: Dict[str, object], namd_systems: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """Return the best matching NAMD system for one GROMACS system."""
    scored = []

    for namd in namd_systems:
        score = match_score(gromacs_system, namd)
        if score > 0:
            scored.append((score, namd))

    if not scored:
        return None

    scored.sort(key=lambda x: (-x[0], str(x[1]["folder"])))

    best_score, best_match = scored[0]

    if len(scored) > 1 and scored[1][0] == best_score:
        print(f"WARNING: tied NAMD matches for {gromacs_system['system_name']}. Using first:")
        for score, namd in scored[:5]:
            print(f"  score={score}  {namd['system_name']}  {namd['folder']}")

    return best_match


# =============================================================================
# COMPARISON ANALYSIS
# =============================================================================

def compare_gromacs_nvt_to_namd_cphmd(
    gromacs_systems: List[Dict[str, object]],
    namd_systems: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Compare initial GROMACS NVT SASA to final NAMD CpHMD SASA.

    Returns:
        comparison_rows
        single_frame_rows
        failure_rows
    """
    comparison_rows = []
    single_frame_rows = []
    failure_rows = []

    for gmx in gromacs_systems:
        namd = match_gromacs_to_namd(gmx, namd_systems)

        if namd is None:
            failure_rows.append({
                "status": "failed",
                "reason": "No matching NAMD CpHMD system found",
                "system_name": gmx["system_name"],
                "bilayer": gmx.get("bilayer", ""),
                "run_name": gmx.get("run_name", ""),
                "target": "ALL",
                "initial_folder": str(gmx["folder"]),
            })
            continue

        for target_name in TARGETS_TO_RUN:
            try:
                initial = calculate_single_frame_sasa(gmx, target_name)
                final = calculate_single_frame_sasa(namd, target_name)

                single_frame_rows.extend([initial, final])

                initial_sasa = initial["SASA"]
                final_sasa = final["SASA"]
                delta_sasa = final_sasa - initial_sasa
                percent_change = (delta_sasa / initial_sasa * 100.0) if initial_sasa != 0 else np.nan

                comparison_rows.append({
                    "status": "complete",
                    "reason": "",

                    "system_name": gmx["system_name"],
                    "matched_namd_system": namd["system_name"],
                    "bilayer": gmx.get("bilayer", ""),
                    "run_name": gmx.get("run_name", ""),
                    "target": target_name,

                    "initial_stage": initial["stage"],
                    "initial_topology_file": initial["topology_file"],
                    "initial_trajectory_file": initial["trajectory_file"],
                    "initial_frame": initial["frame_index"],
                    "initial_time_ns": initial["time_ns"],
                    "initial_SASA": initial_sasa,
                    "initial_SASA_per_atom": initial["SASA_per_atom"],

                    "final_stage": final["stage"],
                    "final_topology_file": final["topology_file"],
                    "final_trajectory_file": final["trajectory_file"],
                    "final_frame": final["frame_index"],
                    "final_time_ns": final["time_ns"],
                    "final_SASA": final_sasa,
                    "final_SASA_per_atom": final["SASA_per_atom"],

                    "delta_SASA": delta_sasa,
                    "percent_change_SASA": percent_change,
                    "delta_SASA_per_atom": final["SASA_per_atom"] - initial["SASA_per_atom"],

                    "n_target_atoms_initial": initial["n_target_atoms"],
                    "n_target_atoms_final": final["n_target_atoms"],

                    "initial_folder": initial["folder"],
                    "initial_topology_path": initial["topology_path"],
                    "initial_trajectory_path": initial["trajectory_path"],

                    "final_folder": final["folder"],
                    "final_topology_path": final["topology_path"],
                    "final_trajectory_path": final["trajectory_path"],

                    "target_selection": initial["target_selection"],
                    "occluder_selection": OCCLUDER_SELECTION,
                    "probe_radius": PROBE_RADIUS,
                    "default_radius": DEFAULT_RADIUS,
                    "n_sphere_points": N_SPHERE_POINTS,
                })

            except Exception as exc:
                failure_rows.append({
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "system_name": gmx["system_name"],
                    "matched_namd_system": namd["system_name"],
                    "bilayer": gmx.get("bilayer", ""),
                    "run_name": gmx.get("run_name", ""),
                    "target": target_name,
                    "initial_folder": str(gmx["folder"]),
                    "final_folder": str(namd["folder"]),
                })

    return comparison_rows, single_frame_rows, failure_rows


# =============================================================================
# OUTPUT
# =============================================================================

def write_csv(path: Path, rows: List[Dict[str, object]], preferred_order: Optional[List[str]] = None) -> None:
    """Write CSV using preferred column order first, then any extra columns."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No rows to write: {path}")
        return

    all_keys = sorted(set().union(*(r.keys() for r in rows)))

    if preferred_order is None:
        fieldnames = all_keys
    else:
        fieldnames = [c for c in preferred_order if c in all_keys]
        fieldnames += [c for c in all_keys if c not in fieldnames]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {path}")


COMPARISON_COLUMN_ORDER = [
    "status",
    "reason",

    "system_name",
    "matched_namd_system",
    "bilayer",
    "run_name",
    "target",

    "initial_stage",
    "initial_topology_file",
    "initial_trajectory_file",
    "initial_frame",
    "initial_time_ns",
    "initial_SASA",
    "initial_SASA_per_atom",

    "final_stage",
    "final_topology_file",
    "final_trajectory_file",
    "final_frame",
    "final_time_ns",
    "final_SASA",
    "final_SASA_per_atom",

    "delta_SASA",
    "percent_change_SASA",
    "delta_SASA_per_atom",

    "n_target_atoms_initial",
    "n_target_atoms_final",

    "initial_folder",
    "final_folder",
    "initial_topology_path",
    "initial_trajectory_path",
    "final_topology_path",
    "final_trajectory_path",

    "target_selection",
    "occluder_selection",
    "probe_radius",
    "default_radius",
    "n_sphere_points",
]


SINGLE_FRAME_COLUMN_ORDER = [
    "engine",
    "system_name",
    "bilayer",
    "run_name",
    "stage",
    "trajectory_type",
    "target",

    "topology_file",
    "trajectory_file",
    "frame_index",
    "n_frames_total",
    "time_ns",

    "SASA",
    "SASA_per_atom",
    "n_target_atoms",
    "n_occluder_atoms",

    "folder",
    "topology_path",
    "trajectory_path",
    "target_selection",
    "occluder_selection",
]


FAILURE_COLUMN_ORDER = [
    "status",
    "reason",
    "system_name",
    "matched_namd_system",
    "bilayer",
    "run_name",
    "target",
    "initial_folder",
    "final_folder",
]


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gromacs_systems = find_gromacs_initial_systems(GROMACS_ROOT)
    namd_systems = find_namd_cphmd_systems(NAMD_ROOT)

    print("=" * 80)
    print("SASA: GROMACS NVT FIRST FRAME -> NAMD CPHMD LAST FRAME")
    print("=" * 80)
    print(f"GROMACS_ROOT: {GROMACS_ROOT}")
    print(f"NAMD_ROOT:    {NAMD_ROOT}")
    print(f"OUTPUT_DIR:   {OUTPUT_DIR}")
    print(f"GROMACS systems found: {len(gromacs_systems)}")
    print(f"NAMD systems found:    {len(namd_systems)}")
    print(f"TARGETS_TO_RUN: {TARGETS_TO_RUN}")
    print(f"OCCLUDER_SELECTION: {OCCLUDER_SELECTION}")
    print("NOTE: SASA is calculated without PBC reconstruction. Use whole/centered trajectories where possible.")

    comparison_rows, single_frame_rows, failure_rows = compare_gromacs_nvt_to_namd_cphmd(
        gromacs_systems=gromacs_systems,
        namd_systems=namd_systems,
    )

    write_csv(
        OUTPUT_DIR / "sasa_nvt_first_to_cphmd_last_delta.csv",
        comparison_rows,
        preferred_order=COMPARISON_COLUMN_ORDER,
    )

    write_csv(
        OUTPUT_DIR / "sasa_single_frame_measurements.csv",
        single_frame_rows,
        preferred_order=SINGLE_FRAME_COLUMN_ORDER,
    )

    write_csv(
        OUTPUT_DIR / "sasa_failures.csv",
        failure_rows,
        preferred_order=FAILURE_COLUMN_ORDER,
    )

    settings = {
        "GROMACS_ROOT": str(GROMACS_ROOT),
        "NAMD_ROOT": str(NAMD_ROOT),
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "comparison": "GROMACS NVT first frame -> NAMD CpHMD last frame",
        "GROMACS_FRAME_INDEX": GROMACS_FRAME_INDEX,
        "NAMD_FRAME_INDEX": NAMD_FRAME_INDEX,
        "TARGETS_TO_RUN": TARGETS_TO_RUN,
        "TARGET_SELECTIONS": TARGET_SELECTIONS,
        "OCCLUDER_SELECTION": OCCLUDER_SELECTION,
        "PROBE_RADIUS": PROBE_RADIUS,
        "N_SPHERE_POINTS": N_SPHERE_POINTS,
        "DEFAULT_RADIUS": DEFAULT_RADIUS,
        "SMALL_RADIUS": SMALL_RADIUS,
        "MAX_DEPTH": MAX_DEPTH,
    }

    with open(OUTPUT_DIR / "sasa_settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    print("\nDone.")
    print(f"Main output: {OUTPUT_DIR / 'sasa_nvt_first_to_cphmd_last_delta.csv'}")


if __name__ == "__main__":
    main()