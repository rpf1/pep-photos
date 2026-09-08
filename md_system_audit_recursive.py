#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
md_system_audit_recursive.py

Purpose
-------
Recursively scan ALL folders below a user-supplied root directory and audit
molecular dynamics system state in a file-layout-agnostic way.

This script does NOT assume a fixed directory structure like:
    bilayer/run_x/peptide_nmol/

Instead, it:
1. Walks every directory under the root
2. Detects candidate simulation folders from the files actually present
3. Extracts as much information as possible from those files
4. Writes one wide summary table to CSV and XLSX

Main goals
----------
- Show what systems exist
- Show what files exist for each system
- Show what stage each system appears to have reached
- Show whether Pull / Relax look complete
- Show approximate final times from logs and/or XTC checks
- Show whether APL outputs exist
- Show enough metadata columns that sorting/filtering is easy

Why this is layout agnostic
---------------------------
The script never hardcodes folder depth or exact path structure.
It only uses:
- recursive folder walking
- detection by actual filenames in each folder
- optional parsing from folder names where possible

Default behaviour
-----------------
- Scans all depth below ROOT_DIR
- Treats any folder containing at least one recognised MD file as a candidate
- Writes:
    audit_md_systems.csv
    audit_md_systems.xlsx

Recommended usage
-----------------
1. Run once with XTC checking disabled if you want a very fast scan
2. Run again with XTC checking enabled if you want stronger final-time validation

Notes
-----
- Log parsing is the primary completion/progress method
- XTC checking is optional because it can be slower
- No files are modified
- No jobs are submitted
- Pure audit only
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from openpyxl.styles import PatternFill, Font
from openpyxl.utils import get_column_letter

# =============================================================================
# USER DEFAULTS
# =============================================================================

DEFAULT_ROOT_DIR = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1"
).resolve()

DEFAULT_OUTPUT_CSV = (
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/audit_md_systems.csv"
)

DEFAULT_XTC_CHECK = False
DEFAULT_GROMACS_CMD = "gmx_mpi"


# =============================================================================
# RECOGNISED FILES
# =============================================================================

RECOGNISED_STAGE_PREFIXES = {
    "box": ["Box.gro"],

    "min": [
        "Min.gro", "Min.log", "Min.tpr", "Min.edr", "Min.cpt",
    ],

    "pull": [
        "Pull.gro", "Pull.log", "Pull.tpr", "Pull.xtc", "Pull.edr", "Pull.cpt",
        "Pull_pullx.xvg", "Pull_pullf.xvg", "Pull.mdp", "Pull.sh",
        "Pull_APL.csv", "Pull_APL.xlsx", "Pull_APL.png",
        "PullMinCOM.gro",
    ],

    "relax": [
        "Relax.gro", "Relax.log", "Relax.tpr", "Relax.xtc", "Relax.edr", "Relax.cpt",
        "Relax.mdp", "Relax.sh",
        "Relax_APL.csv", "Relax_APL.xlsx", "Relax_APL.png",
    ],

    # Placeholder stage.
    # Edit these filenames later when your CpHMD workflow is finalized.
    "cphmd": [
        "CpHMD.gro", "CpHMD.log", "CpHMD.tpr", "CpHMD.xtc", "CpHMD.edr", "CpHMD.cpt",
        "CpHMD.mdp", "CpHMD.sh",
        "CpHMD_APL.csv", "CpHMD_APL.xlsx", "CpHMD_APL.png",
        "CpHMD_stats.csv", "CpHMD_stats.xlsx", "CpHMD_stats.png",
    ],
}

CANONICAL_MD_FILENAMES = {
    name.lower()
    for group in RECOGNISED_STAGE_PREFIXES.values()
    for name in group
}

# also recognise these patterns generically
GENERIC_PATTERNS = [
    r"^topol.*\.top$",
    r"^index.*\.ndx$",
    r"^specialindex.*\.ndx$",
    r"^positions\.dat$",
    r"^.*pullx.*\.xvg$",
    r"^.*pullf.*\.xvg$",
    r"^.*apl.*\.(csv|xlsx|png)$",
    r"^slurm.*\.log$",
    r"^submit.*\.sh$",
]


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class StageInfo:
    exists: bool = False
    log_exists: bool = False
    tpr_exists: bool = False
    gro_exists: bool = False
    xtc_exists: bool = False
    cpt_exists: bool = False
    edr_exists: bool = False
    mdp_exists: bool = False
    sh_exists: bool = False
    finished_mdrun: Optional[bool] = None
    fatal_error: Optional[bool] = None
    parsed_final_time_ps: Optional[float] = None
    parsed_final_time_ns: Optional[float] = None
    xtc_final_time_ps: Optional[float] = None
    xtc_final_time_ns: Optional[float] = None
    best_final_time_ps: Optional[float] = None
    best_final_time_ns: Optional[float] = None


# =============================================================================
# BASIC HELPERS
# =============================================================================

def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def file_exists(folder: Path, filename: str) -> bool:
    return (folder / filename).exists()


def find_first_existing(folder: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = folder / name
        if p.exists():
            return p
    return None


def normalise_bool(x: Optional[bool]) -> str:
    if x is True:
        return "yes"
    if x is False:
        return "no"
    return ""


def ps_to_ns(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value / 1000.0


# =============================================================================
# DETECTION
# =============================================================================

def is_recognised_file(name: str) -> bool:
    lower = name.lower()

    if lower in CANONICAL_MD_FILENAMES:
        return True

    for pattern in GENERIC_PATTERNS:
        if re.match(pattern, name, flags=re.IGNORECASE):
            return True

    return False


def folder_has_md_signal(folder: Path) -> bool:
    try:
        for item in folder.iterdir():
            if item.is_file() and is_recognised_file(item.name):
                return True
    except PermissionError:
        return False
    return False


def collect_candidate_folders(root_dir: Path) -> List[Path]:
    candidates = []
    for p in root_dir.rglob("*"):
        if not p.is_dir():
            continue
        if folder_has_md_signal(p):
            candidates.append(p)
    return sorted(set(candidates))


# =============================================================================
# FILE LISTING
# =============================================================================

def list_files(folder: Path) -> List[Path]:
    try:
        return sorted([p for p in folder.iterdir() if p.is_file()])
    except PermissionError:
        return []


def file_names(folder: Path) -> List[str]:
    return [p.name for p in list_files(folder)]


def count_matching_files(folder: Path, pattern: str) -> int:
    return sum(1 for p in list_files(folder) if re.match(pattern, p.name, flags=re.IGNORECASE))


# =============================================================================
# LOG PARSING
# =============================================================================

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            return path.read_text(errors="replace")
        except Exception:
            return ""


def parse_finished_mdrun(log_text: str) -> Optional[bool]:
    if not log_text:
        return None
    return "Finished mdrun" in log_text


def parse_fatal_error(log_text: str) -> Optional[bool]:
    if not log_text:
        return None
    return "Fatal error:" in log_text


def parse_last_time_ps_from_log(log_text: str) -> Optional[float]:
    """
    Robust attempt to parse the final reported time in ps from a GROMACS log.

    Handles patterns like:
        Step           Time
        4000000        100000.000

    Also scans repeated step/time blocks and takes the last valid time found.
    """
    if not log_text:
        return None

    lines = log_text.splitlines()
    candidate_times: List[float] = []

    for i, line in enumerate(lines):
        if "Step" in line and "Time" in line:
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                parts = next_line.split()
                if len(parts) >= 2:
                    t = safe_float(parts[1])
                    if t is not None:
                        candidate_times.append(t)

    # fallback: capture explicit "time" appearances if needed
    if not candidate_times:
        patterns = [
            r"\btime\s+([0-9eE+\-.]+)",
            r"\bTime\s+([0-9eE+\-.]+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, log_text)
            for m in matches:
                t = safe_float(m)
                if t is not None:
                    candidate_times.append(t)

    if not candidate_times:
        return None

    return candidate_times[-1]


# =============================================================================
# XTC CHECKING
# =============================================================================

def parse_last_time_ps_from_gmx_check(text: str) -> Optional[float]:
    patterns = [
        r"Last frame\s+\d+\s+time\s+([0-9eE+\-.]+)",
        r"Last\s+frame\s+\d+\s+time\s+([0-9eE+\-.]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            t = safe_float(matches[-1])
            if t is not None:
                return t
    return None


def xtc_final_time_ps(xtc_path: Path, gromacs_cmd: str) -> Tuple[Optional[float], str]:
    result = run_cmd([gromacs_cmd, "check", "-f", str(xtc_path)], cwd=xtc_path.parent)
    combined = "\n".join(part for part in [result.stdout, result.stderr] if part)

    if result.returncode != 0:
        return None, combined.strip()

    value = parse_last_time_ps_from_gmx_check(combined)
    if value is None:
        return None, "could not parse final time from gmx check output"

    return value, ""


# =============================================================================
# STAGE AUDIT
# =============================================================================

def audit_stage(folder: Path, stage_name: str, xtc_check: bool, gromacs_cmd: str) -> StageInfo:
    info = StageInfo()

    log_path = folder / f"{stage_name}.log"
    tpr_path = folder / f"{stage_name}.tpr"
    gro_path = folder / f"{stage_name}.gro"
    xtc_path = folder / f"{stage_name}.xtc"
    cpt_path = folder / f"{stage_name}.cpt"
    edr_path = folder / f"{stage_name}.edr"
    mdp_path = folder / f"{stage_name}.mdp"
    sh_path = folder / f"{stage_name}.sh"

    info.log_exists = log_path.exists()
    info.tpr_exists = tpr_path.exists()
    info.gro_exists = gro_path.exists()
    info.xtc_exists = xtc_path.exists()
    info.cpt_exists = cpt_path.exists()
    info.edr_exists = edr_path.exists()
    info.mdp_exists = mdp_path.exists()
    info.sh_exists = sh_path.exists()

    info.exists = any([
        info.log_exists, info.tpr_exists, info.gro_exists, info.xtc_exists,
        info.cpt_exists, info.edr_exists, info.mdp_exists, info.sh_exists
    ])

    if info.log_exists:
        log_text = read_text(log_path)
        info.finished_mdrun = parse_finished_mdrun(log_text)
        info.fatal_error = parse_fatal_error(log_text)
        info.parsed_final_time_ps = parse_last_time_ps_from_log(log_text)
        info.parsed_final_time_ns = ps_to_ns(info.parsed_final_time_ps)

    if xtc_check and info.xtc_exists:
        xtc_ps, _ = xtc_final_time_ps(xtc_path, gromacs_cmd=gromacs_cmd)
        info.xtc_final_time_ps = xtc_ps
        info.xtc_final_time_ns = ps_to_ns(xtc_ps)

    candidates = [x for x in [info.xtc_final_time_ps, info.parsed_final_time_ps] if x is not None]
    if candidates:
        info.best_final_time_ps = max(candidates)
        info.best_final_time_ns = ps_to_ns(info.best_final_time_ps)

    return info


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_folder(
    has_box: bool,
    min_info: StageInfo,
    pull_info: StageInfo,
    relax_info: StageInfo,
    cphmd_info: StageInfo,
    has_pull_apl: bool,
    has_relax_apl: bool,
    has_cphmd_analysis: bool,
) -> Tuple[str, str]:

    if cphmd_info.finished_mdrun is True:
        if has_cphmd_analysis:
            return "cphmd_complete_analysis_done", "CpHMD log contains Finished mdrun and CpHMD analysis output exists"
        return "cphmd_complete", "CpHMD log contains Finished mdrun"

    if cphmd_info.exists:
        if cphmd_info.fatal_error is True:
            return "cphmd_failed", "CpHMD files exist and CpHMD.log contains Fatal error"
        if cphmd_info.best_final_time_ns is not None:
            return "cphmd_in_progress_or_incomplete", f"CpHMD files exist; best final CpHMD time = {cphmd_info.best_final_time_ns:.3f} ns"
        return "cphmd_present_unknown_progress", "CpHMD files exist but final CpHMD time could not be parsed"

    if relax_info.finished_mdrun is True:
        if has_relax_apl:
            return "relax_complete_apl_done", "Relax log contains Finished mdrun and Relax_APL output exists"
        return "relax_complete", "Relax log contains Finished mdrun"

    if relax_info.exists:
        if relax_info.fatal_error is True:
            return "relax_failed", "Relax files exist and Relax.log contains Fatal error"
        if relax_info.best_final_time_ns is not None:
            return "relax_in_progress_or_incomplete", f"Relax files exist; best final relax time = {relax_info.best_final_time_ns:.3f} ns"
        return "relax_present_unknown_progress", "Relax files exist but final relax time could not be parsed"

    if pull_info.finished_mdrun is True:
        if has_pull_apl:
            return "pull_complete_apl_done_waiting_relax", "Pull log contains Finished mdrun and Pull_APL output exists"
        return "pull_complete_waiting_relax", "Pull log contains Finished mdrun"

    if pull_info.exists:
        if pull_info.fatal_error is True:
            return "pull_failed", "Pull files exist and Pull.log contains Fatal error"
        if pull_info.best_final_time_ns is not None:
            return "pull_in_progress_or_incomplete", f"Pull files exist; best final pull time = {pull_info.best_final_time_ns:.3f} ns"
        return "pull_present_unknown_progress", "Pull files exist but final pull time could not be parsed"

    if min_info.exists:
        return "min_stage_only", "Minimisation files exist but no Pull files found"

    if has_box:
        return "box_only", "Box.gro exists but no minimisation files found"

    return "detected_misc_md_folder", "Recognised MD-related files exist but no canonical stage classification matched"


# =============================================================================
# OPTIONAL PATH PARSING
# =============================================================================

def extract_sequence_from_row(row: pd.Series) -> str:
    rel_path = str(row.get("relative_path", ""))
    folder_name = str(row.get("folder_name", ""))

    parts = Path(rel_path).parts

    # Preptide/AKRIRKLIKKIFKKI
    # Preptide/AKRIRKLIKKIFKKI/anything/deeper
    for i, part in enumerate(parts):
        if part.lower() == "preptide" and i + 1 < len(parts):
            return parts[i + 1]

    # Bilayer/run_0/AKRIRKLIKKIFKKI_80
    # Bilayer/run_0/Nk-Nfex-Nfex_80
    for part in reversed(parts):
        m = re.match(r"(.+?)[_-](\d+)$", part)
        if m:
            return m.group(1)

    if folder_name:
        m = re.match(r"(.+?)[_-](\d+)$", folder_name)
        if m:
            return m.group(1)

    return folder_name


def extract_bilayer_from_row(row: pd.Series) -> str:
    rel_path = str(row.get("relative_path", ""))
    parts = Path(rel_path).parts

    if not parts:
        return ""

    if parts[0].lower() == "preptide":
        return "Preptide"

    return parts[0]


def classify_molecule_type(sequence: str) -> str:
    if "-" in str(sequence):
        return "Peptoid"
    return "Peptide"


def status_rank(status: str) -> int:
    """
    Higher number = more advanced / more useful final status.
    Used to collapse duplicate recursive rows.
    """
    status = str(status)

    ranks = {
        "cphmd_complete": 100,
        "cphmd_complete_analysis_done": 95,
        "cphmd_in_progress_or_incomplete": 90,
        "cphmd_present_unknown_progress": 85,
        "cphmd_failed": 80,

        "relax_complete_apl_done": 70,
        "relax_complete": 65,
        "relax_in_progress_or_incomplete": 60,
        "relax_present_unknown_progress": 55,
        "relax_failed": 50,

        "pull_complete_apl_done_waiting_relax": 40,
        "pull_complete_waiting_relax": 35,
        "pull_in_progress_or_incomplete": 30,
        "pull_present_unknown_progress": 25,
        "pull_failed": 20,

        "min_stage_only": 10,
        "box_only": 5,
        "detected_misc_md_folder": 1,
    }

    return ranks.get(status, 0)


def broad_state_from_status(status: str) -> str:
    status = str(status).lower()

    if "failed" in status:
        return "Failed"

    if status.startswith("cphmd_complete"):
        return "CpHMD Passed"

    if status.startswith("relax_complete"):
        return "Relax Passed"

    if status.startswith("pull") and "failed" not in status:
        return "Still Running / Pull Stage"

    if status.startswith("relax") and "failed" not in status:
        return "Still Running / Relax Stage"

    if status.startswith("cphmd") and "failed" not in status:
        return "Still Running / CpHMD Stage"

    return "Other / Unknown"


def guess_sequence_from_path(folder: Path, root_dir: Path) -> str:
    """
    Extracts sequence from paths like:

    Preptide/AKRIRKLIKKIFKKI
    Preptide/AKRIRKLIKKIFKKI/run_0
    Preptide/AKRIRKLIKKIFKKI/some/deeper/folder

    For normal bilayer folders like:

    POPC60CHOL40_W_WF/run_0/AKRIRKLIKKIFKKI_80

    it extracts AKRIRKLIKKIFKKI from the final system folder.
    """
    rel_parts = folder.relative_to(root_dir).parts

    for i, part in enumerate(rel_parts):
        if part.lower() == "preptide" and i + 1 < len(rel_parts):
            return rel_parts[i + 1]

    folder_name = folder.name
    return guess_sequence_from_folder_name(folder_name)


def parse_run_number_from_path(folder: Path) -> Optional[int]:
    for part in folder.parts[::-1]:
        m = re.match(r"run[_-]?(\d+)$", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def parse_nmol_from_folder_name(name: str) -> Optional[int]:
    """
    Tries to parse patterns like:
        peptide_80
        sequence-80
        sequence_213
    Uses final integer token after underscore or hyphen.
    """
    m = re.search(r"[_-](\d+)$", name)
    if m:
        return int(m.group(1))
    return None


def guess_sequence_from_folder_name(name: str) -> str:
    m = re.search(r"(.+?)[_-]\d+$", name)
    if m:
        return m.group(1)
    return name


# =============================================================================
# ROW BUILDING
# =============================================================================

def build_row(folder: Path, root_dir: Path, xtc_check: bool, gromacs_cmd: str) -> Dict[str, object]:
    names = file_names(folder)
    names_lower = {x.lower() for x in names}

    has_box_gro = "box.gro" in names_lower
    has_topol_top = any(re.match(r"^topol.*\.top$", x, flags=re.IGNORECASE) for x in names)
    has_positions_dat = any(x.lower() == "positions.dat" for x in names)
    has_index_ndx = any(re.match(r"^index.*\.ndx$", x, flags=re.IGNORECASE) for x in names)
    has_specialindex_ndx = any(re.match(r"^specialindex.*\.ndx$", x, flags=re.IGNORECASE) for x in names)
    has_pull_apl_csv = any(x.lower() == "pull_apl.csv" for x in names)
    has_relax_apl_csv = any(x.lower() == "relax_apl.csv" for x in names)
    has_pull_apl_any = any(re.match(r"^pull_apl\.(csv|xlsx|png)$", x, flags=re.IGNORECASE) for x in names)
    has_relax_apl_any = any(re.match(r"^relax_apl\.(csv|xlsx|png)$", x, flags=re.IGNORECASE) for x in names)

    min_info = audit_stage(folder, "Min", xtc_check=xtc_check, gromacs_cmd=gromacs_cmd)
    pull_info = audit_stage(folder, "Pull", xtc_check=xtc_check, gromacs_cmd=gromacs_cmd)
    relax_info = audit_stage(folder, "Relax", xtc_check=xtc_check, gromacs_cmd=gromacs_cmd)
    cphmd_info = audit_stage(folder, "CpHMD", xtc_check=xtc_check, gromacs_cmd=gromacs_cmd)
    
    has_cphmd_analysis_any = any(
        re.match(r"^cphmd.*\.(csv|xlsx|png)$", x, flags=re.IGNORECASE)
        for x in names
    )

    status, status_reason = classify_folder(
        has_box=has_box_gro,
        min_info=min_info,
        pull_info=pull_info,
        relax_info=relax_info,
        cphmd_info=cphmd_info,
        has_pull_apl=has_pull_apl_any,
        has_relax_apl=has_relax_apl_any,
        has_cphmd_analysis=has_cphmd_analysis_any,
    )

    rel_path = folder.relative_to(root_dir)
    path_parts = folder.parts
    run_number = parse_run_number_from_path(folder)
    folder_name = folder.name
    guessed_nmol = parse_nmol_from_folder_name(folder_name)
    guessed_sequence = guess_sequence_from_path(folder, root_dir)
    is_preptide_path = path_contains_part(folder, "Preptide")
    bilayer_guess = guess_bilayer_from_relative_path(rel_path)

    row: Dict[str, object] = {
        # identity
        "root_dir": str(root_dir),
        "folder_path": str(folder),
        "relative_path": str(rel_path),
        "folder_name": folder_name,
        "parent_folder_name": folder.parent.name if folder.parent else "",
        "path_depth_from_root": len(rel_path.parts),
        "run_number_guess": run_number,
        "nmol_guess": guessed_nmol,
        "is_preptide_path": is_preptide_path,
        "bilayer_guess": bilayer_guess,

        # raw folder/file summary
        "file_count_total": len(names),
        "recognised_md_file_count": sum(is_recognised_file(x) for x in names),
        "all_files": " | ".join(names),

        # setup files
        "has_Box_gro": has_box_gro,
        "has_topol_top": has_topol_top,
        "has_positions_dat": has_positions_dat,
        "has_index_ndx": has_index_ndx,
        "has_SpecialIndex_ndx": has_specialindex_ndx,

        # min files
        "has_Min_any": min_info.exists,
        "has_Min_log": min_info.log_exists,
        "has_Min_tpr": min_info.tpr_exists,
        "has_Min_gro": min_info.gro_exists,
        "has_Min_xtc": min_info.xtc_exists,
        "has_Min_cpt": min_info.cpt_exists,
        "has_Min_edr": min_info.edr_exists,
        "has_Min_mdp": min_info.mdp_exists,
        "has_Min_sh": min_info.sh_exists,
        "Min_finished_mdrun": normalise_bool(min_info.finished_mdrun),
        "Min_fatal_error": normalise_bool(min_info.fatal_error),
        "Min_final_time_ps_log": min_info.parsed_final_time_ps,
        "Min_final_time_ns_log": min_info.parsed_final_time_ns,
        "Min_final_time_ps_xtc": min_info.xtc_final_time_ps,
        "Min_final_time_ns_xtc": min_info.xtc_final_time_ns,
        "Min_final_time_ps_best": min_info.best_final_time_ps,
        "Min_final_time_ns_best": min_info.best_final_time_ns,

        # pull files
        "has_Pull_any": pull_info.exists,
        "has_Pull_log": pull_info.log_exists,
        "has_Pull_tpr": pull_info.tpr_exists,
        "has_Pull_gro": pull_info.gro_exists,
        "has_Pull_xtc": pull_info.xtc_exists,
        "has_Pull_cpt": pull_info.cpt_exists,
        "has_Pull_edr": pull_info.edr_exists,
        "has_Pull_mdp": pull_info.mdp_exists,
        "has_Pull_sh": pull_info.sh_exists,
        "has_Pull_pullx_xvg": file_exists(folder, "Pull_pullx.xvg"),
        "has_Pull_pullf_xvg": file_exists(folder, "Pull_pullf.xvg"),
        "has_Pull_APL_csv": has_pull_apl_csv,
        "has_Pull_APL_any": has_pull_apl_any,
        "has_PullMinCOM_gro": file_exists(folder, "PullMinCOM.gro"),
        "Pull_finished_mdrun": normalise_bool(pull_info.finished_mdrun),
        "Pull_fatal_error": normalise_bool(pull_info.fatal_error),
        "Pull_final_time_ps_log": pull_info.parsed_final_time_ps,
        "Pull_final_time_ns_log": pull_info.parsed_final_time_ns,
        "Pull_final_time_ps_xtc": pull_info.xtc_final_time_ps,
        "Pull_final_time_ns_xtc": pull_info.xtc_final_time_ns,
        "Pull_final_time_ps_best": pull_info.best_final_time_ps,
        "Pull_final_time_ns_best": pull_info.best_final_time_ns,

        # relax files
        "has_Relax_any": relax_info.exists,
        "has_Relax_log": relax_info.log_exists,
        "has_Relax_tpr": relax_info.tpr_exists,
        "has_Relax_gro": relax_info.gro_exists,
        "has_Relax_xtc": relax_info.xtc_exists,
        "has_Relax_cpt": relax_info.cpt_exists,
        "has_Relax_edr": relax_info.edr_exists,
        "has_Relax_mdp": relax_info.mdp_exists,
        "has_Relax_sh": relax_info.sh_exists,
        "has_Relax_APL_csv": has_relax_apl_csv,
        "has_Relax_APL_any": has_relax_apl_any,
        "Relax_finished_mdrun": normalise_bool(relax_info.finished_mdrun),
        "Relax_fatal_error": normalise_bool(relax_info.fatal_error),
        "Relax_final_time_ps_log": relax_info.parsed_final_time_ps,
        "Relax_final_time_ns_log": relax_info.parsed_final_time_ns,
        "Relax_final_time_ps_xtc": relax_info.xtc_final_time_ps,
        "Relax_final_time_ns_xtc": relax_info.xtc_final_time_ns,
        "Relax_final_time_ps_best": relax_info.best_final_time_ps,
        "Relax_final_time_ns_best": relax_info.best_final_time_ns,
        
        # cphmd files
        "has_CpHMD_any": cphmd_info.exists,
        "has_CpHMD_log": cphmd_info.log_exists,
        "has_CpHMD_tpr": cphmd_info.tpr_exists,
        "has_CpHMD_gro": cphmd_info.gro_exists,
        "has_CpHMD_xtc": cphmd_info.xtc_exists,
        "has_CpHMD_cpt": cphmd_info.cpt_exists,
        "has_CpHMD_edr": cphmd_info.edr_exists,
        "has_CpHMD_mdp": cphmd_info.mdp_exists,
        "has_CpHMD_sh": cphmd_info.sh_exists,
        "has_CpHMD_analysis_any": has_cphmd_analysis_any,
        "CpHMD_finished_mdrun": normalise_bool(cphmd_info.finished_mdrun),
        "CpHMD_fatal_error": normalise_bool(cphmd_info.fatal_error),
        "CpHMD_final_time_ps_log": cphmd_info.parsed_final_time_ps,
        "CpHMD_final_time_ns_log": cphmd_info.parsed_final_time_ns,
        "CpHMD_final_time_ps_xtc": cphmd_info.xtc_final_time_ps,
        "CpHMD_final_time_ns_xtc": cphmd_info.xtc_final_time_ns,
        "CpHMD_final_time_ps_best": cphmd_info.best_final_time_ps,
        "CpHMD_final_time_ns_best": cphmd_info.best_final_time_ns,
        
        # generic extras
        "submit_sh_count": count_matching_files(folder, r"^submit.*\.sh$"),
        "slurm_log_count": count_matching_files(folder, r"^slurm.*\.log$"),
        "any_csv_count": count_matching_files(folder, r"^.*\.csv$"),
        "any_xlsx_count": count_matching_files(folder, r"^.*\.xlsx$"),
        "any_png_count": count_matching_files(folder, r"^.*\.png$"),
        "any_xtc_count": count_matching_files(folder, r"^.*\.xtc$"),
        "any_gro_count": count_matching_files(folder, r"^.*\.gro$"),
        "any_log_count": count_matching_files(folder, r"^.*\.log$"),

        # classification
        "status": status,
        "status_reason": status_reason,
    }

    return row


# =============================================================================
# OUTPUT
# =============================================================================

def make_summary_counts_df(condensed_df: pd.DataFrame) -> pd.DataFrame:
    if condensed_df.empty:
        return pd.DataFrame(columns=[
            "Molecule Type",
            "Total",
            "Relax Passed",
            "Still Running",
            "Failed",
            "CpHMD Passed",
            "Relax Passed / Total",
            "Still Running / Total",
            "Failed / Total",
            "CpHMD Passed / Total",
        ])

    rows = []

    for mol_type in ["Peptide", "Peptoid"]:
        sub = condensed_df[condensed_df["Molecule Type"] == mol_type].copy()
        total = len(sub)

        relax_passed = sub["status"].astype(str).str.startswith("relax_complete").sum()
        cphmd_passed = sub["status"].astype(str).str.startswith("cphmd_complete").sum()
        failed = sub["status"].astype(str).str.contains("failed", case=False, na=False).sum()

        still_running = sub["broad_state"].isin([
            "Still Running / Pull Stage",
            "Still Running / Relax Stage",
            "Still Running / CpHMD Stage",
        ]).sum()

        rows.append({
            "Molecule Type": mol_type,
            "Total": total,
            "Relax Passed": relax_passed,
            "Still Running": still_running,
            "Failed": failed,
            "CpHMD Passed": cphmd_passed,
            "Relax Passed / Total": f"{relax_passed}/{total}",
            "Still Running / Total": f"{still_running}/{total}",
            "Failed / Total": f"{failed}/{total}",
            "CpHMD Passed / Total": f"{cphmd_passed}/{total}",
        })

    return pd.DataFrame(rows)


def path_contains_part(folder: Path, target: str) -> bool:
    return any(part.lower() == target.lower() for part in folder.parts)


def guess_bilayer_from_relative_path(rel_path: Path) -> str:
    parts = list(rel_path.parts)

    if not parts:
        return ""

    if parts[0].lower() == "preptide":
        return "Preptide"

    return parts[0]


def extract_sequence_for_condensed(row: pd.Series) -> str:
    rel_path = str(row.get("relative_path", ""))
    folder_name = str(row.get("folder_name", ""))

    parts = Path(rel_path).parts

    # Case 1:
    # Preptide/AKRIRKLIKKIFKKI
    # Preptide/AKRIRKLIKKIFKKI/anything/deeper
    for i, part in enumerate(parts):
        if part.lower() == "preptide" and i + 1 < len(parts):
            return parts[i + 1]

    # Case 2:
    # POPC60CHOL40_W_WF/run_0/AKRIRKLIKKIFKKI_80
    # Extract sequence from final system folder
    if folder_name:
        m = re.match(r"(.+?)[_-]\d+$", folder_name)
        if m:
            return m.group(1)
        return folder_name

    return ""


def extract_bilayer_for_condensed(row: pd.Series) -> str:
    rel_path = str(row.get("relative_path", ""))
    parts = Path(rel_path).parts

    if not parts:
        return ""

    if parts[0].lower() == "preptide":
        return "Preptide"

    return parts[0]


def make_condensed_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "Sequence",
            "Molecule Type",
            "Bilayer",
            "Pull.gro Exists",
            "Relax.gro Exists",
            "CpHMD.gro Exists",
            "status",
            "status_reason",
            "broad_state",
            "source_folder",
        ])

    work = df.copy()

    work["Sequence"] = work.apply(extract_sequence_from_row, axis=1)
    work["Bilayer"] = work.apply(extract_bilayer_from_row, axis=1)
    work["Molecule Type"] = work["Sequence"].apply(classify_molecule_type)
    work["status_rank"] = work["status"].apply(status_rank)
    work["broad_state"] = work["status"].apply(broad_state_from_status)

    # Prefer rows that are actual system folders with clearer stage files.
    work["stage_file_score"] = (
        work.get("has_Pull_gro", False).astype(int)
        + work.get("has_Relax_gro", False).astype(int) * 2
        + work.get("has_CpHMD_gro", False).astype(int) * 3
        + work.get("has_Pull_log", False).astype(int)
        + work.get("has_Relax_log", False).astype(int) * 2
        + work.get("has_CpHMD_log", False).astype(int) * 3
    )

    work = work.sort_values(
        ["Sequence", "Bilayer", "status_rank", "stage_file_score", "relative_path"],
        ascending=[True, True, False, False, True],
    )

    # One row per sequence + bilayer.
    work = work.drop_duplicates(subset=["Sequence", "Bilayer"], keep="first")

    condensed = pd.DataFrame()
    condensed["Sequence"] = work["Sequence"]
    condensed["Molecule Type"] = work["Molecule Type"]
    condensed["Bilayer"] = work["Bilayer"]
    condensed["Pull.gro Exists"] = work["has_Pull_gro"] if "has_Pull_gro" in work.columns else False
    condensed["Relax.gro Exists"] = work["has_Relax_gro"] if "has_Relax_gro" in work.columns else False
    condensed["CpHMD.gro Exists"] = work["has_CpHMD_gro"] if "has_CpHMD_gro" in work.columns else False
    condensed["status"] = work["status"]
    condensed["status_reason"] = work["status_reason"]
    condensed["broad_state"] = work["broad_state"]
    condensed["source_folder"] = work["folder_path"]

    return condensed.reset_index(drop=True)


def style_workbook(writer: pd.ExcelWriter) -> None:
    true_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    false_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    header_fill = PatternFill(start_color="D9EAF7", end_color="D9EAF7", fill_type="solid")

    workbook = writer.book

    for ws in workbook.worksheets:
        ws.freeze_panes = "A2"

        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill

        for row in ws.iter_rows():
            for cell in row:
                value = cell.value

                if value is True or value == "True" or value == "yes":
                    cell.fill = true_fill

                elif value is False or value == "False" or value == "no":
                    cell.fill = false_fill

        for column_cells in ws.columns:
            max_length = 0
            column_letter = get_column_letter(column_cells[0].column)

            for cell in column_cells:
                if cell.value is not None:
                    max_length = max(max_length, len(str(cell.value)))

            ws.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 45)

        ws.auto_filter.ref = ws.dimensions
        

def save_outputs(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)

    xlsx_path = csv_path.with_suffix(".xlsx")

    preptide_df = df[
        df["relative_path"].astype(str).str.contains(r"(^|/)Preptide(/|$)", case=False, regex=True)
    ].copy() if "relative_path" in df.columns else pd.DataFrame()

    condensed_df = make_condensed_df(df)
    summary_counts_df = make_summary_counts_df(condensed_df)

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All_Audit", index=False)
            preptide_df.to_excel(writer, sheet_name="Preptide", index=False)
            condensed_df.to_excel(writer, sheet_name="Condensed", index=False)
            summary_counts_df.to_excel(writer, sheet_name="Summary_Counts", index=False)

            style_workbook(writer)

    except Exception as exc:
        print(f"Could not write XLSX: {exc}")


# =============================================================================
# COLUMN GUIDE
# =============================================================================

def print_column_guide() -> None:
    guide = """
COLUMN REASONING
================

Identity columns
----------------
root_dir
    Root scan location used for this audit.

folder_path
    Absolute path to the audited folder.

relative_path
    Easier to read than full absolute path and useful for sorting.def make_summary_counts_df(condensed_df: pd.DataFrame) -> pd.DataFrame:
    if condensed_df.empty:
        return pd.DataFrame(columns=[
            "Molecule Type",
            "Total",
            "Relax Passed",
            "Still Running",
            "Failed",
            "CpHMD Passed",
            "Relax Passed / Total",
            "Still Running / Total",
            "Failed / Total",
            "CpHMD Passed / Total",
        ])

    rows = []

    for mol_type in ["Peptide", "Peptoid"]:
        sub = condensed_df[condensed_df["Molecule Type"] == mol_type].copy()
        total = len(sub)

        relax_passed = sub["status"].astype(str).str.startswith("relax_complete").sum()
        cphmd_passed = sub["status"].astype(str).str.startswith("cphmd_complete").sum()
        failed = sub["status"].astype(str).str.contains("failed", case=False, na=False).sum()

        still_running = sub["broad_state"].isin([
            "Still Running / Pull Stage",
            "Still Running / Relax Stage",
            "Still Running / CpHMD Stage",
        ]).sum()

        rows.append({
            "Molecule Type": mol_type,
            "Total": total,
            "Relax Passed": relax_passed,
            "Still Running": still_running,
            "Failed": failed,
            "CpHMD Passed": cphmd_passed,
            "Relax Passed / Total": f"{relax_passed}/{total}",
            "Still Running / Total": f"{still_running}/{total}",
            "Failed / Total": f"{failed}/{total}",
            "CpHMD Passed / Total": f"{cphmd_passed}/{total}",
        })

    return pd.DataFrame(rows)

folder_name
    Quick visual system identifier.

parent_folder_name
    Helps group related folders.

path_depth_from_root
    Useful for checking whether certain systems are deeper than expected.

run_number_guess
    Parsed from path if a folder like run_0 / run-0 exists.

nmol_guess
    Parsed from folder name if it ends with _80 / -80 / etc.

sequence_guess
    Best-effort guess of system identifier from folder name.

Raw file summary
----------------
file_count_total
    Total number of files in the folder.

recognised_md_file_count
    Number of files matching known MD-related names/patterns.

all_files
    Full file list for manual inspection and debugging.

Setup columns
-------------
has_Box_gro
    Whether a built simulation box exists.

has_topol_top
    Whether topology exists.

has_positions_dat
    Whether insertion positions were generated.

has_index_ndx / has_SpecialIndex_ndx
    Whether index files were prepared.

Stage presence columns
----------------------
has_Min_any / has_Pull_any / has_Relax_any
    Whether any file for that stage exists at all.

Per-stage file columns
----------------------
has_<Stage>_log
    Job log exists.

has_<Stage>_tpr
    Run input exists.

has_<Stage>_gro
    Structure output exists.

has_<Stage>_xtc
    Trajectory exists.

has_<Stage>_cpt
    Checkpoint exists.

has_<Stage>_edr
    Energy file exists.

has_<Stage>_mdp
    MDP file exists.

has_<Stage>_sh
    Stage shell script exists.

Completion / error columns
--------------------------
<Stage>_finished_mdrun
    Whether "Finished mdrun" was found in that stage log.

<Stage>_fatal_error
    Whether "Fatal error:" was found in that stage log.

Time columns
------------
<Stage>_final_time_ps_log / ns_log
    Final time parsed from the stage log.

<Stage>_final_time_ps_xtc / ns_xtc
    Final time obtained from gmx check on the XTC, if enabled.

<Stage>_final_time_ps_best / ns_best
    Best available final time, using XTC if available, otherwise log.

APL / analysis columns
----------------------
has_Pull_APL_csv / has_Relax_APL_csv
    Whether the expected APL CSV outputs exist.

has_Pull_APL_any / has_Relax_APL_any
    Broader check for CSV/XLSX/PNG APL outputs.

has_PullMinCOM_gro
    Whether frame extraction for relax setup appears to have happened.

Generic convenience columns
---------------------------
submit_sh_count
    Number of submit scripts in the folder.

slurm_log_count
    Number of slurm logs in the folder.

any_csv_count / any_xlsx_count / any_png_count
    Quick summary of downstream analysis outputs.

any_xtc_count / any_gro_count / any_log_count
    Quick summary of simulation-stage file density.

Classification columns
----------------------
status
    Simple machine-readable category for folder state.

status_reason
    Human-readable explanation of the status.

Why this helps with missing analysis rows
-----------------------------------------
If a system is missing from your analysis CSV, this audit tells you whether:
- the folder exists at all
- Pull or Relax was ever created
- Pull or Relax completed
- APL outputs exist
- the system likely failed
- the system exists but was probably skipped by your analysis script
"""
    print(guide)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively audit MD system folders and write CSV/XLSX summary."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT_DIR),
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV filename or full path.",
    )
    parser.add_argument(
        "--xtc-check",
        action="store_true",
        default=DEFAULT_XTC_CHECK,
        help="Run `gmx_mpi check -f` on XTC files to extract final times.",
    )
    parser.add_argument(
        "--gmx",
        type=str,
        default=DEFAULT_GROMACS_CMD,
        help="GROMACS command to use, default: gmx_mpi",
    )
    parser.add_argument(
        "--show-column-guide",
        action="store_true",
        help="Print explanation of all output columns before scanning.",
    )

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root_dir / out_path

    if args.show_column_guide:
        print_column_guide()

    print(f"Scanning root: {root_dir}")
    print(f"Output CSV  : {out_path}")
    print(f"Output XLSX : {out_path.with_suffix('.xlsx')}")
    print(f"XTC check   : {args.xtc_check}")

    candidates = collect_candidate_folders(root_dir)
    print(f"Candidate folders found: {len(candidates)}")

    rows: List[Dict[str, object]] = []

    for i, folder in enumerate(candidates, start=1):
        print(f"[{i}/{len(candidates)}] Auditing: {folder}")
        row = build_row(
            folder=folder,
            root_dir=root_dir,
            xtc_check=args.xtc_check,
            gromacs_cmd=args.gmx,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=[
            "root_dir", "folder_path", "relative_path", "folder_name",
            "status", "status_reason"
        ])

    # Useful default sort
    sort_cols = [c for c in [
        "status",
        "run_number_guess",
        "relative_path",
    ] if c in df.columns]

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    save_outputs(df, out_path)

    print("\nDone.")
    print(f"CSV : {out_path}")
    print(f"XLSX: {out_path.with_suffix('.xlsx')}")


if __name__ == "__main__":
    main()
