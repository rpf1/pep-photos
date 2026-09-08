#!/usr/bin/env python3
"""
Read-only PFPeptoids GROMACS performance audit.

Scans a project tree for Pull.log and Relax.log files, extracts GROMACS
performance/configuration information, and supplements it with atom counts,
box dimensions and charges from a matching stage TPR when MDAnalysis can read
it. A matching GRO is used as a fallback for atom count and box dimensions.

The script never edits simulation inputs, trajectories, logs or database files.
It creates only a new timestamped audit directory beneath --output-parent.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
)
DEFAULT_OUTPUT_PARENT = DEFAULT_ROOT / "analysis" / "simulation_performance_audits"

BILAYER_MARKERS = (
    "POPC", "POPS", "POPE", "POPG", "DPPC", "DTPC", "CHOL"
)

PEPTOID_RESNAMES = {
    "Na", "Nab", "Ni", "Nl", "Nm", "Nv",
    "Nn", "Nq", "Ns", "Nse", "Nt", "NmO",
    "Nd", "Ne",
    "Nf", "Nfn", "Nfe", "Nfex", "Nw", "Nwe", "Ny",
    "Nfe4Cl", "Nfe4Br", "Nfnaph", "NfeC4", "NfeB4", "Nfnap",
    "Nk", "Nke", "Nr", "Nkeqm",
}

PEPTIDE_RESNAMES = {
    "ALAC", "ALAN", "ALA", "VALC", "VALN", "VAL",
    "ILEC", "ILEN", "ILE", "LEUC", "LEUN", "LEU",
    "METC", "METN", "MET", "CYSC", "CYSN", "CYS",
    "PROC", "PRON", "PRO",
    "SERC", "SERN", "SER", "THRC", "THRN", "THR",
    "ASNC", "ASNN", "ASN", "GLNC", "GLNN", "GLN",
    "HSEC", "HSEN", "HSE",
    "ASPC", "ASPN", "ASP", "GLUC", "GLUN", "GLU",
    "PHEC", "PHEN", "PHE", "TYRC", "TYRN", "TYR",
    "TRPC", "TRPN", "TRP",
    "LYSC", "LYSN", "LYS", "ARGC", "ARGN", "ARG",
}

POLYMER_RESNAMES = PEPTOID_RESNAMES | PEPTIDE_RESNAMES

DETAIL_COLUMNS = [
    "peptoid",
    "nmol",
    "bilayer",
    "stage",
    "cycle",
    "run_name",
    "status",
    "log_path",
    "system_directory",
    "log_modified_utc",
    "log_size_mb",
    "box_x_nm",
    "box_y_nm",
    "box_z_nm",
    "box_volume_nm3",
    "peptoid_charge_per_molecule_e",
    "total_polymer_charge_e",
    "total_system_charge_e",
    "charge_source",
    "number_of_atoms",
    "polymer_atom_count",
    "planned_simulation_ns",
    "simulation_ns_completed",
    "start_time_ps",
    "end_time_ps",
    "configured_nsteps",
    "configured_dt_ps",
    "simulation_ns_per_day",
    "median_ns_per_day_across_log_segments",
    "minimum_ns_per_day_across_log_segments",
    "maximum_ns_per_day_across_log_segments",
    "performance_records",
    "hours_per_simulated_ns",
    "estimated_wall_hours_for_completed_ns",
    "mpi_processes",
    "openmp_threads_per_rank",
    "estimated_cores_used",
    "cpu_brand",
    "hostname",
    "warning_count",
    "lincs_warning_count",
    "fatal_error_detected",
    "structure_source",
    "metadata_source",
    "parse_notes",
]


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit all PFPeptoids Pull.log and Relax.log files read-only."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Project tree to scan (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=DEFAULT_OUTPUT_PARENT,
        help=f"Parent for timestamped results (default: {DEFAULT_OUTPUT_PARENT})",
    )
    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=0.80,
        help=(
            "Flag completed runs below this fraction of the median for the same "
            "stage, bilayer and core count (default: 0.80)."
        ),
    )
    parser.add_argument(
        "--minimum-comparison-group",
        type=int,
        default=3,
        help="Minimum completed runs required for relative slow-run flagging.",
    )
    parser.add_argument(
        "--skip-tpr-metadata",
        action="store_true",
        help="Skip MDAnalysis TPR loading; use logs/GRO files only.",
    )
    return parser.parse_args()


def first_match(pattern: str, text: str, flags: int = re.MULTILINE) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def last_number(pattern: str, text: str, cast: type = float) -> Any | None:
    matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
    if not matches:
        return None
    try:
        return cast(matches[-1])
    except (TypeError, ValueError):
        return None


def safe_round(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    if abs(number) < 10 ** (-(digits + 1)):
        number = 0.0
    return round(number, digits)


def path_context(log_path: Path) -> dict[str, Any]:
    parts = log_path.parts
    peptoid = None
    nmol = None
    peptoid_pattern = re.compile(r"^(.+)-(.*)-(.+)_([0-9]+)$")

    for part in reversed(parts[:-1]):
        match = peptoid_pattern.match(part)
        if match:
            peptoid = part.rsplit("_", 1)[0]
            nmol = int(match.group(4))
            break

    bilayer = None
    for part in reversed(parts[:-1]):
        if any(marker in part.upper() for marker in BILAYER_MARKERS):
            bilayer = part
            break

    cycle = None
    run_name = None
    cycle_pattern = re.compile(
        r"(?:run[_-]?)?(?:ML[_-]?)?cycle[_-]?0*([0-9]+)", re.IGNORECASE
    )
    for part in parts:
        match = cycle_pattern.search(part)
        if match:
            cycle = int(match.group(1))
            run_name = part

    return {
        "peptoid": peptoid,
        "nmol": nmol,
        "bilayer": bilayer,
        "cycle": cycle,
        "run_name": run_name,
    }


def find_stage_file(directory: Path, stage: str, suffix: str) -> Path | None:
    exact_names = [
        f"{stage}{suffix}",
        f"{stage.lower()}{suffix}",
        f"{stage.upper()}{suffix}",
    ]
    for name in exact_names:
        candidate = directory / name
        if candidate.is_file():
            return candidate

    suffix_lower = suffix.lower()
    stage_lower = stage.lower()
    candidates = sorted(
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == suffix_lower
        and stage_lower in p.stem.lower()
    )
    return candidates[0] if candidates else None


def parse_gro_metadata(gro_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "number_of_atoms": None,
        "box_x_nm": None,
        "box_y_nm": None,
        "box_z_nm": None,
        "box_volume_nm3": None,
        "metadata_source": str(gro_path),
        "notes": [],
    }
    try:
        with gro_path.open("r", encoding="utf-8", errors="replace") as handle:
            title = handle.readline()
            atom_count_line = handle.readline().strip()
            if not title or not atom_count_line:
                raise ValueError("GRO header is incomplete")
            atom_count = int(atom_count_line)
            result["number_of_atoms"] = atom_count

            last_line = ""
            for line in handle:
                last_line = line.strip()

        values = [float(value) for value in last_line.split()]
        if len(values) == 3:
            box_x, box_y, box_z = values
            volume = box_x * box_y * box_z
        elif len(values) == 9:
            v1 = (values[0], values[3], values[4])
            v2 = (values[5], values[1], values[6])
            v3 = (values[7], values[8], values[2])
            box_x = math.sqrt(sum(x * x for x in v1))
            box_y = math.sqrt(sum(x * x for x in v2))
            box_z = math.sqrt(sum(x * x for x in v3))
            volume = abs(
                v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
                - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
                + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
            )
        else:
            raise ValueError(f"expected 3 or 9 box values, found {len(values)}")

        result.update(
            box_x_nm=safe_round(box_x),
            box_y_nm=safe_round(box_y),
            box_z_nm=safe_round(box_z),
            box_volume_nm3=safe_round(volume),
        )
    except Exception as exc:
        result["notes"].append(f"GRO metadata failed: {exc}")
    return result


def parse_tpr_metadata(tpr_path: Path, nmol: int | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "number_of_atoms": None,
        "polymer_atom_count": None,
        "box_x_nm": None,
        "box_y_nm": None,
        "box_z_nm": None,
        "box_volume_nm3": None,
        "peptoid_charge_per_molecule_e": None,
        "total_polymer_charge_e": None,
        "total_system_charge_e": None,
        "charge_source": None,
        "metadata_source": str(tpr_path),
        "notes": [],
    }

    try:
        import MDAnalysis as mda
    except Exception as exc:
        result["notes"].append(f"MDAnalysis unavailable: {exc}")
        return result

    try:
        universe = mda.Universe(str(tpr_path))
        result["number_of_atoms"] = int(universe.atoms.n_atoms)

        dimensions = universe.dimensions
        if dimensions is not None and len(dimensions) >= 3:
            # MDAnalysis reports lengths in Angstrom; GROMACS/GRO uses nm.
            result["box_x_nm"] = safe_round(float(dimensions[0]) / 10.0)
            result["box_y_nm"] = safe_round(float(dimensions[1]) / 10.0)
            result["box_z_nm"] = safe_round(float(dimensions[2]) / 10.0)
            try:
                result["box_volume_nm3"] = safe_round(
                    float(universe.trajectory.ts.volume) / 1000.0
                )
            except Exception:
                result["box_volume_nm3"] = safe_round(
                    result["box_x_nm"]
                    * result["box_y_nm"]
                    * result["box_z_nm"]
                )

        atom_resnames = [str(name) for name in universe.atoms.resnames]
        polymer_indices = [
            index
            for index, residue_name in enumerate(atom_resnames)
            if residue_name in POLYMER_RESNAMES
        ]
        result["polymer_atom_count"] = len(polymer_indices)

        try:
            charges = universe.atoms.charges
            total_system_charge = sum(float(charge) for charge in charges)
            total_polymer_charge = sum(float(charges[index]) for index in polymer_indices)

            result["total_system_charge_e"] = safe_round(total_system_charge, 5)
            result["total_polymer_charge_e"] = safe_round(total_polymer_charge, 5)
            if nmol and nmol > 0 and polymer_indices:
                result["peptoid_charge_per_molecule_e"] = safe_round(
                    total_polymer_charge / nmol, 5
                )
            result["charge_source"] = "matching TPR parsed by MDAnalysis"
        except Exception as exc:
            result["notes"].append(f"TPR charge extraction failed: {exc}")

    except Exception as exc:
        result["notes"].append(f"TPR metadata failed: {exc}")

    return result


def extract_step_time_records(lines: list[str]) -> list[tuple[int, float]]:
    records: list[tuple[int, float]] = []
    header_pattern = re.compile(r"^\s*Step\s+Time(?:\s|$)", re.IGNORECASE)
    data_pattern = re.compile(rf"^\s*(-?\d+)\s+({FLOAT_RE})(?:\s|$)")

    for index, line in enumerate(lines):
        if not header_pattern.search(line):
            continue
        for offset in range(1, 7):
            if index + offset >= len(lines):
                break
            candidate = lines[index + offset]
            match = data_pattern.match(candidate)
            if match:
                records.append((int(match.group(1)), float(match.group(2))))
                break
            if candidate.strip() and not set(candidate.strip()) <= {"-", " "}:
                # Continue through one or two headings/separators, but do not
                # wander far into an unrelated section.
                continue
    return records


def parse_log(log_path: Path, skip_tpr_metadata: bool) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    stage = log_path.stem.capitalize()
    context = path_context(log_path)
    notes: list[str] = []

    mpi_processes = last_number(
        r"Using\s+(\d+)\s+MPI process", text, int
    )
    openmp_threads = last_number(
        r"Using\s+(\d+)\s+OpenMP thread", text, int
    )
    cores_used = (
        mpi_processes * openmp_threads
        if mpi_processes is not None and openmp_threads is not None
        else None
    )

    performance_pairs = [
        (float(ns_day), float(hours_ns))
        for ns_day, hours_ns in re.findall(
            rf"^\s*Performance:\s+({FLOAT_RE})\s+({FLOAT_RE})",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
    ]
    ns_day_values = [pair[0] for pair in performance_pairs]
    latest_ns_day = ns_day_values[-1] if ns_day_values else None
    latest_hours_ns = performance_pairs[-1][1] if performance_pairs else None

    configured_nsteps = last_number(
        r"^\s*nsteps\s*=\s*(-?\d+)", text, int
    )
    configured_dt_ps = last_number(
        rf"^\s*dt\s*=\s*({FLOAT_RE})", text, float
    )
    planned_ns = None
    if (
        configured_nsteps is not None
        and configured_dt_ps is not None
        and configured_nsteps >= 0
    ):
        planned_ns = configured_nsteps * configured_dt_ps / 1000.0

    step_time_records = extract_step_time_records(lines)
    start_time_ps = None
    end_time_ps = None
    completed_ns = None
    if step_time_records:
        times = [record[1] for record in step_time_records]
        start_time_ps = min(times)
        end_time_ps = max(times)
        completed_ns = max(0.0, end_time_ps - start_time_ps) / 1000.0
        if completed_ns == 0 and end_time_ps > 0:
            completed_ns = end_time_ps / 1000.0
    elif performance_pairs and planned_ns is not None:
        completed_ns = planned_ns
        notes.append("completed duration inferred from configured nsteps × dt")

    warning_count = len(re.findall(r"^\s*WARNING\b", text, re.MULTILINE))
    lincs_warning_count = len(re.findall(r"LINCS WARNING", text, re.IGNORECASE))
    fatal_error = bool(
        re.search(
            r"Fatal error|Segmentation fault|Out Of Memory|oom-kill|CANCELLED|TIME LIMIT",
            text,
            re.IGNORECASE,
        )
    )
    finished = bool(re.search(r"Finished mdrun", text, re.IGNORECASE))

    if fatal_error:
        status = "failed"
    elif finished:
        status = "completed"
    elif performance_pairs:
        status = "completed_or_clean_stop"
    else:
        status = "running_or_incomplete"

    cpu_brand = first_match(r"^\s*CPU brand:\s*(.+)$", text)
    if cpu_brand is None:
        cpu_brand = first_match(r"^\s*Brand:\s*(.+)$", text)

    hostname = first_match(r"Hardware detected on host\s+([^:]+):", text)
    if hostname is None:
        hostname = first_match(r"^\s*Host(?:name)?:\s*(\S+)", text)

    number_of_atoms = last_number(
        r"^\s*(\d+)\s+atoms(?:\s|$)", text, int
    )
    total_system_charge = last_number(
        rf"System has non-zero total charge:\s*({FLOAT_RE})", text, float
    )

    metadata: dict[str, Any] = {
        "number_of_atoms": number_of_atoms,
        "polymer_atom_count": None,
        "box_x_nm": None,
        "box_y_nm": None,
        "box_z_nm": None,
        "box_volume_nm3": None,
        "peptoid_charge_per_molecule_e": None,
        "total_polymer_charge_e": None,
        "total_system_charge_e": total_system_charge,
        "charge_source": "GROMACS log total-system charge only" if total_system_charge is not None else None,
        "metadata_source": None,
        "notes": [],
    }

    tpr_path = find_stage_file(log_path.parent, stage, ".tpr")
    gro_path = find_stage_file(log_path.parent, stage, ".gro")

    if tpr_path is not None and not skip_tpr_metadata:
        tpr_metadata = parse_tpr_metadata(tpr_path, context["nmol"])
        for key, value in tpr_metadata.items():
            if key == "notes":
                metadata["notes"].extend(value)
            elif value is not None:
                metadata[key] = value

    if (
        gro_path is not None
        and (
            metadata["number_of_atoms"] is None
            or metadata["box_x_nm"] is None
            or metadata["box_y_nm"] is None
            or metadata["box_z_nm"] is None
        )
    ):
        gro_metadata = parse_gro_metadata(gro_path)
        for key in (
            "number_of_atoms", "box_x_nm", "box_y_nm", "box_z_nm", "box_volume_nm3"
        ):
            if metadata.get(key) is None and gro_metadata.get(key) is not None:
                metadata[key] = gro_metadata[key]
        if metadata["metadata_source"] is None:
            metadata["metadata_source"] = gro_metadata["metadata_source"]
        metadata["notes"].extend(gro_metadata["notes"])

    if metadata["metadata_source"] is None:
        metadata["metadata_source"] = "GROMACS log only"
        if tpr_path is None:
            notes.append(f"matching {stage}.tpr not found")
        if gro_path is None:
            notes.append(f"matching {stage}.gro not found")

    notes.extend(metadata["notes"])
    estimated_wall_hours = None
    if completed_ns is not None and latest_hours_ns is not None:
        estimated_wall_hours = completed_ns * latest_hours_ns

    stat = log_path.stat()
    modified_utc = datetime.fromtimestamp(
        stat.st_mtime, tz=timezone.utc
    ).isoformat(timespec="seconds")

    return {
        "peptoid": context["peptoid"],
        "nmol": context["nmol"],
        "bilayer": context["bilayer"],
        "stage": stage,
        "cycle": context["cycle"],
        "run_name": context["run_name"],
        "status": status,
        "log_path": str(log_path.resolve()),
        "system_directory": str(log_path.parent.resolve()),
        "log_modified_utc": modified_utc,
        "log_size_mb": safe_round(stat.st_size / (1024 * 1024), 4),
        "box_x_nm": metadata["box_x_nm"],
        "box_y_nm": metadata["box_y_nm"],
        "box_z_nm": metadata["box_z_nm"],
        "box_volume_nm3": metadata["box_volume_nm3"],
        "peptoid_charge_per_molecule_e": metadata["peptoid_charge_per_molecule_e"],
        "total_polymer_charge_e": metadata["total_polymer_charge_e"],
        "total_system_charge_e": metadata["total_system_charge_e"],
        "charge_source": metadata["charge_source"],
        "number_of_atoms": metadata["number_of_atoms"],
        "polymer_atom_count": metadata["polymer_atom_count"],
        "planned_simulation_ns": safe_round(planned_ns),
        "simulation_ns_completed": safe_round(completed_ns),
        "start_time_ps": safe_round(start_time_ps),
        "end_time_ps": safe_round(end_time_ps),
        "configured_nsteps": configured_nsteps,
        "configured_dt_ps": configured_dt_ps,
        "simulation_ns_per_day": safe_round(latest_ns_day),
        "median_ns_per_day_across_log_segments": safe_round(
            statistics.median(ns_day_values) if ns_day_values else None
        ),
        "minimum_ns_per_day_across_log_segments": safe_round(
            min(ns_day_values) if ns_day_values else None
        ),
        "maximum_ns_per_day_across_log_segments": safe_round(
            max(ns_day_values) if ns_day_values else None
        ),
        "performance_records": len(performance_pairs),
        "hours_per_simulated_ns": safe_round(latest_hours_ns),
        "estimated_wall_hours_for_completed_ns": safe_round(estimated_wall_hours),
        "mpi_processes": mpi_processes,
        "openmp_threads_per_rank": openmp_threads,
        "estimated_cores_used": cores_used,
        "cpu_brand": cpu_brand,
        "hostname": hostname,
        "warning_count": warning_count,
        "lincs_warning_count": lincs_warning_count,
        "fatal_error_detected": fatal_error,
        "structure_source": str(tpr_path or gro_path) if (tpr_path or gro_path) else None,
        "metadata_source": metadata["metadata_source"],
        "parse_notes": "; ".join(dict.fromkeys(note for note in notes if note)),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def build_summaries(
    rows: list[dict[str, Any]],
    slow_threshold: float,
    minimum_group_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["bilayer"], row["estimated_cores_used"])].append(row)

    summaries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        stage, bilayer, cores = key
        performances = numeric_values(group_rows, "simulation_ns_per_day")
        atoms = numeric_values(group_rows, "number_of_atoms")
        volumes = numeric_values(group_rows, "box_volume_nm3")
        median_performance = statistics.median(performances) if performances else None

        summaries.append({
            "stage": stage,
            "bilayer": bilayer,
            "estimated_cores_used": cores,
            "log_count": len(group_rows),
            "completed_performance_count": len(performances),
            "median_ns_per_day": safe_round(median_performance),
            "mean_ns_per_day": safe_round(statistics.fmean(performances) if performances else None),
            "minimum_ns_per_day": safe_round(min(performances) if performances else None),
            "maximum_ns_per_day": safe_round(max(performances) if performances else None),
            "median_number_of_atoms": safe_round(statistics.median(atoms) if atoms else None),
            "median_box_volume_nm3": safe_round(statistics.median(volumes) if volumes else None),
            "failed_count": sum(row["status"] == "failed" for row in group_rows),
            "incomplete_count": sum(row["status"] == "running_or_incomplete" for row in group_rows),
            "warning_total": sum(int(row["warning_count"] or 0) for row in group_rows),
            "lincs_warning_total": sum(int(row["lincs_warning_count"] or 0) for row in group_rows),
        })

        for row in group_rows:
            reasons: list[str] = []
            relative = None
            performance = row.get("simulation_ns_per_day")

            if (
                performance is not None
                and median_performance is not None
                and len(performances) >= minimum_group_size
                and median_performance > 0
            ):
                relative = float(performance) / median_performance
                if relative < slow_threshold:
                    reasons.append(
                        f"performance is {relative:.1%} of comparable-group median"
                    )

            if row["status"] == "failed":
                reasons.append("fatal/failure text detected")
            elif row["status"] == "running_or_incomplete":
                reasons.append("no completed performance record")
            if int(row["warning_count"] or 0) > 0:
                reasons.append(f"{row['warning_count']} GROMACS warning(s)")
            if int(row["lincs_warning_count"] or 0) > 0:
                reasons.append(f"{row['lincs_warning_count']} LINCS warning(s)")
            if row["number_of_atoms"] is None:
                reasons.append("atom count unavailable")
            if row["box_volume_nm3"] is None:
                reasons.append("box dimensions unavailable")

            if reasons:
                issue_row = dict(row)
                issue_row["comparable_group_median_ns_per_day"] = safe_round(median_performance)
                issue_row["fraction_of_group_median"] = safe_round(relative)
                issue_row["audit_flags"] = "; ".join(reasons)
                issues.append(issue_row)

    summaries.sort(
        key=lambda row: (
            str(row["stage"]),
            str(row["bilayer"]),
            row["estimated_cores_used"] if row["estimated_cores_used"] is not None else -1,
        )
    )
    issues.sort(
        key=lambda row: (
            row["simulation_ns_per_day"] is None,
            row["simulation_ns_per_day"] if row["simulation_ns_per_day"] is not None else float("inf"),
            str(row["log_path"]),
        )
    )
    return summaries, issues


def write_report(
    path: Path,
    root: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    status_counts = Counter(row["status"] for row in rows)
    cycles = Counter(row["cycle"] for row in rows)
    stages = Counter(row["stage"] for row in rows)
    performances = numeric_values(rows, "simulation_ns_per_day")

    lines = [
        "PFPEPTOIDS GROMACS PERFORMANCE AUDIT",
        "=" * 80,
        f"Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Read-only scan root: {root.resolve()}",
        f"Log files inspected: {len(rows)}",
        f"Stages: {dict(sorted(stages.items(), key=lambda item: str(item[0])))}",
        f"Cycles: {dict(sorted(cycles.items(), key=lambda item: str(item[0])))}",
        f"Statuses: {dict(sorted(status_counts.items()))}",
        f"Logs with performance values: {len(performances)}",
        f"Overall median ns/day: {safe_round(statistics.median(performances) if performances else None)}",
        f"Flagged rows: {len(issues)}",
        "",
        "COMPARABLE-GROUP SUMMARY",
        "-" * 80,
    ]

    for summary in summaries:
        lines.append(
            " | ".join([
                f"stage={summary['stage']}",
                f"bilayer={summary['bilayer']}",
                f"cores={summary['estimated_cores_used']}",
                f"n={summary['completed_performance_count']}/{summary['log_count']}",
                f"median={summary['median_ns_per_day']} ns/day",
                f"range={summary['minimum_ns_per_day']}–{summary['maximum_ns_per_day']} ns/day",
                f"median_atoms={summary['median_number_of_atoms']}",
            ])
        )

    completed_rows = [row for row in rows if row["simulation_ns_per_day"] is not None]
    completed_rows.sort(key=lambda row: float(row["simulation_ns_per_day"]))
    lines.extend(["", "TWENTY SLOWEST COMPLETED LOG RECORDS", "-" * 80])
    for row in completed_rows[:20]:
        lines.append(
            " | ".join([
                f"{row['simulation_ns_per_day']} ns/day",
                f"{row['peptoid']}",
                f"{row['bilayer']}",
                f"{row['stage']}",
                f"cycle={row['cycle']}",
                f"cores={row['estimated_cores_used']}",
                f"atoms={row['number_of_atoms']}",
                row["log_path"],
            ])
        )

    lines.extend([
        "",
        "INTERPRETATION RULES",
        "-" * 80,
        "Compare ns/day only between equivalent stages, similar bilayers/system sizes,",
        "the same core layout and preferably the same CPU/node type.",
        "The latest Performance line is used when a log contains multiple run segments.",
        "Charge per peptoid is reported only when a matching TPR can be parsed and",
        "the polymer residues can be identified; no charge is guessed from the sequence.",
        "A running log normally lacks a final Performance line and is not automatically",
        "evidence of poor performance.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_parent = args.output_parent.expanduser().resolve()

    if not root.is_dir():
        print(f"ERROR: scan root does not exist: {root}", file=sys.stderr)
        return 2
    if not (0 < args.slow_threshold <= 1.0):
        print("ERROR: --slow-threshold must be greater than 0 and at most 1", file=sys.stderr)
        return 2

    log_paths = sorted(
        path for path in root.rglob("*.log")
        if path.name.lower() in {"pull.log", "relax.log"}
    )
    if not log_paths:
        print(f"ERROR: no Pull.log or Relax.log files found beneath {root}", file=sys.stderr)
        return 3

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_parent / f"audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"Read-only scan root: {root}")
    print(f"Logs found: {len(log_paths)}")
    print(f"Output directory: {output_dir}")

    rows: list[dict[str, Any]] = []
    for index, log_path in enumerate(log_paths, start=1):
        print(f"[{index:>4}/{len(log_paths)}] {log_path}")
        try:
            rows.append(parse_log(log_path, args.skip_tpr_metadata))
        except Exception as exc:
            context = path_context(log_path)
            stat = log_path.stat()
            failed_row = {column: None for column in DETAIL_COLUMNS}
            failed_row.update(
                peptoid=context["peptoid"],
                nmol=context["nmol"],
                bilayer=context["bilayer"],
                stage=log_path.stem.capitalize(),
                cycle=context["cycle"],
                run_name=context["run_name"],
                status="audit_parse_failed",
                log_path=str(log_path.resolve()),
                system_directory=str(log_path.parent.resolve()),
                log_modified_utc=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
                log_size_mb=safe_round(stat.st_size / (1024 * 1024), 4),
                fatal_error_detected=None,
                parse_notes=f"Unhandled parser error: {type(exc).__name__}: {exc}",
            )
            rows.append(failed_row)

    rows.sort(
        key=lambda row: (
            row["cycle"] if row["cycle"] is not None else 999999,
            str(row["peptoid"]),
            str(row["bilayer"]),
            str(row["stage"]),
            str(row["log_path"]),
        )
    )

    summaries, issues = build_summaries(
        rows,
        slow_threshold=args.slow_threshold,
        minimum_group_size=args.minimum_comparison_group,
    )

    detailed_path = output_dir / "simulation_performance_detailed.csv"
    summary_path = output_dir / "simulation_performance_group_summary.csv"
    issues_path = output_dir / "simulation_performance_flags.csv"
    report_path = output_dir / "simulation_performance_report.txt"

    write_csv(detailed_path, rows, DETAIL_COLUMNS)

    summary_columns = [
        "stage", "bilayer", "estimated_cores_used", "log_count",
        "completed_performance_count", "median_ns_per_day", "mean_ns_per_day",
        "minimum_ns_per_day", "maximum_ns_per_day", "median_number_of_atoms",
        "median_box_volume_nm3", "failed_count", "incomplete_count",
        "warning_total", "lincs_warning_total",
    ]
    write_csv(summary_path, summaries, summary_columns)

    issue_columns = DETAIL_COLUMNS + [
        "comparable_group_median_ns_per_day",
        "fraction_of_group_median",
        "audit_flags",
    ]
    write_csv(issues_path, issues, issue_columns)
    write_report(report_path, root, rows, summaries, issues)

    print("")
    print("AUDIT COMPLETE")
    print(f"Detailed records: {detailed_path}")
    print(f"Grouped summary:  {summary_path}")
    print(f"Flags/outliers:   {issues_path}")
    print(f"Text report:      {report_path}")
    print("No simulation files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


