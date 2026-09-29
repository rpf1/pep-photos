#!/usr/bin/env python3
"""Consolidated read-only PFPeptoids Cycle-1/Cycle-2 campaign audit.

Reads, without modifying:
  * active_learning.db (opened mode=ro and query_only=ON)
  * masterdataanalysis.csv
  * live run_ML_cycle_01 and run_ML_cycle_02 directories
  * Pull/Relax MDP and log files (never trajectories)
  * current Slurm queue and recent Slurm accounting, when available
  * workflow.py and model_search_config.json

Writes exactly two files into a new timestamped audit directory:
  * campaign_audit_report.txt
  * campaign_stage_state.csv

No simulation, analysis, synchronization, model training, selection, or Slurm
submission is performed.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ROOT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
)
DEFAULT_OUTPUT_PARENT = DEFAULT_ROOT / "analysis" / "campaign_state_audits"
EXPECTED_WORKFLOW_SHA256 = (
    "20ea9479f173ed29eed5ebb7347db8d0f0e3a80fe2e913237f28a50f11ad1ef8"
)

BACTERIAL = "POPE75POPG25_W_WF"
MAMMALIAN = "POPC60CHOL40_W_WF"
SYSTEMS = (BACTERIAL, MAMMALIAN)
STAGES = ("Pull", "Relax")
CANONICAL_BY_CYCLE = {1: set("ABCDE"), 2: set("ABCDEFG")}

CORRECTED_B1 = "Nf-Nfe-Nm"
INVALID_B1 = "Nf-Nf-Nkeqm"
CORRECTED_CE = {"Nf-Nfex-Nk", "Nf-Ni-Nwe", "Nf-Nl-Nwe", "Nf-Nw-Nl"}
EXPECTED_D2 = {
    "Nn-Nw-Nwe", "Nn-Nwe-Nwe", "Nn-NfeB4-Nwe", "Nwe-Nw-Na",
    "Nn-Nfnap-Nwe", "Nk-Nw-Nwe", "Nk-Nk-Nwe", "Nk-Nfnap-Nk",
    "Nwe-Nwe-Na", "Nk-Nw-Nk", "Nse-Na-Nfn", "Nse-Nfn-Nfn",
    "Nse-Nm-Nfn", "Ne-Nm-Nfn", "Nm-Nse-Nfn",
}

TRUE_TEXT = {"1", "true", "yes", "y", "complete", "completed", "ready", "analysis_ready", "done"}
FALSE_TEXT = {"0", "false", "no", "n", "incomplete", "failed", "pending", "running", "selected"}

CSV_FIELDS = [
    "cycle", "canonical", "strategies", "peptoid_id", "system", "stage",
    "directory", "directory_exists", "mdp_path", "mdp_exists", "dt_ps",
    "nsteps", "planned_ns", "parameter_valid", "tpr_exists", "log_exists",
    "checkpoint_exists", "checkpoint_size_bytes", "max_step", "max_time_ps",
    "completed_ns", "progress_percent", "finished_marker", "fatal_marker",
    "last_performance_ns_day", "last_checkpoint_step", "stage_state",
    "master_rows", "master_valid_rows", "master_analysis_valid",
    "preferred_observation", "db_queue_statuses", "acquisition_state",
    "current_job_ids", "current_job_states", "recent_job_history",
    "job_name_path_mismatches",
    "recommended_action", "action_reason", "log_modified_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument(
        "--sacct-start",
        default="2026-08-01",
        help="Start date for one read-only sacct query (default: 2026-08-01).",
    )
    parser.add_argument("--no-slurm", action="store_true")
    return parser.parse_args()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def canonical_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"_80(?:\b.*)?$", "", text)
    return text


def truth(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return None


def finite_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def int_or_none(value: Any) -> int | None:
    number = finite_float(value)
    return int(number) if number is not None else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_column(
    columns: Sequence[str], exact: Sequence[str], contains: Sequence[Sequence[str]] = ()
) -> str | None:
    by_norm = {norm(column): column for column in columns}
    for candidate in exact:
        found = by_norm.get(norm(candidate))
        if found:
            return found
    for parts in contains:
        for column in columns:
            normalized = norm(column)
            if all(norm(part) in normalized for part in parts):
                return column
    return None


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def sqlite_tables(con: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def sqlite_columns(con: sqlite3.Connection, table: str) -> list[str]:
    safe = table.replace('"', '""')
    return [row[1] for row in con.execute(f'PRAGMA table_info("{safe}")')]


def sqlite_all(con: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    safe = table.replace('"', '""')
    cursor = con.execute(f'SELECT * FROM "{safe}"')
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def system_from_text(*values: Any) -> str | None:
    text = " ".join(str(value or "") for value in values)
    for system in SYSTEMS:
        if system.lower() in text.lower():
            return system
    return None


def context_from_path(path: str | Path) -> tuple[int | None, str | None, str | None]:
    text = str(path)
    cycle_match = re.search(r"run_ML_cycle_0*([12])", text, re.IGNORECASE)
    id_match = re.search(r"/([^/]+)_80(?:/|$)", text)
    return (
        int(cycle_match.group(1)) if cycle_match else None,
        canonical_id(id_match.group(1)) if id_match else None,
        system_from_text(text),
    )


def parse_mdp(path: Path) -> dict[str, Any]:
    result = {"dt_ps": None, "nsteps": None, "notes": []}
    if not path.is_file():
        return result
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.split(";", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[norm(key)] = value.strip()
    except OSError as exc:
        result["notes"].append(f"mdp_read_error={exc}")
        return result
    result["dt_ps"] = finite_float(values.get("dt"))
    result["nsteps"] = int_or_none(values.get("nsteps") or values.get("nstep"))
    return result


STEP_TIME_RE = re.compile(
    r"^[ \t]*Step[ \t]+Time[ \t]*\r?\n[ \t]*([0-9]+)[ \t]+([-+0-9.Ee]+)",
    re.MULTILINE,
)


def parse_gromacs_log(path: Path) -> dict[str, Any]:
    result = {
        "max_step": None,
        "max_time_ps": None,
        "last_checkpoint_step": None,
        "finished_marker": False,
        "fatal_marker": False,
        "last_performance_ns_day": None,
        "log_modified_utc": "",
        "log_dt_ps": None,
        "log_nsteps": None,
        "notes": [],
    }
    if not path.is_file():
        return result
    try:
        stat = path.stat()
        result["log_modified_utc"] = datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat()
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result["notes"].append(f"log_read_error={exc}")
        return result

    pairs: list[tuple[int, float]] = []
    for step_text, time_text in STEP_TIME_RE.findall(text):
        try:
            pairs.append((int(step_text), float(time_text)))
        except ValueError:
            pass
    if pairs:
        result["max_step"] = max(step for step, _ in pairs)
        result["max_time_ps"] = max(time_ps for _, time_ps in pairs)

    checkpoints = [int(value) for value in re.findall(r"Writing checkpoint, step\s+([0-9]+)", text)]
    if checkpoints:
        result["last_checkpoint_step"] = checkpoints[-1]

    performances = [
        float(value)
        for value in re.findall(r"Performance:\s+([-+0-9.Ee]+)", text)
        if finite_float(value) is not None
    ]
    if performances:
        result["last_performance_ns_day"] = performances[-1]

    result["finished_marker"] = bool(
        re.search(r"Finished mdrun|GROMACS reminds you", text, re.IGNORECASE)
    )
    result["fatal_marker"] = bool(
        re.search(
            r"Fatal error|Inconsistency in user input|Segmentation fault|MPI_ABORT|constraint error",
            text,
            re.IGNORECASE,
        )
    )

    dt_values = re.findall(r"^[ \t]*dt[ \t]*=[ \t]*([-+0-9.Ee]+)", text, re.MULTILINE)
    nstep_values = re.findall(r"^[ \t]*nsteps[ \t]*=[ \t]*([0-9]+)", text, re.MULTILINE)
    if dt_values:
        result["log_dt_ps"] = finite_float(dt_values[-1])
    if nstep_values:
        result["log_nsteps"] = int_or_none(nstep_values[-1])
    return result


def run_command(command: Sequence[str], timeout: int = 90) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(command), text=True, capture_output=True, check=False, timeout=timeout
        )
        return completed.returncode, completed.stdout, completed.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)


def job_context(job_name: str, workdir: str) -> tuple[int | None, str | None, str | None, str | None]:
    cycle, peptoid_id, system = context_from_path(workdir)
    stage_match = re.search(r"(?:_80-|-)(Pull|Relax)(?:-|$)", job_name, re.IGNORECASE)
    stage = stage_match.group(1).title() if stage_match else None
    if peptoid_id is None:
        id_match = re.search(r"([A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)_80", job_name)
        peptoid_id = canonical_id(id_match.group(1)) if id_match else None
    if system is None:
        system = system_from_text(job_name)
    if cycle is None:
        cycle_match = re.search(r"runML_cycle_0*([12])", job_name, re.IGNORECASE)
        cycle = int(cycle_match.group(1)) if cycle_match else None
    return cycle, peptoid_id, system, stage


def job_name_path_mismatch(job_name: str, workdir: str) -> str:
    path_cycle, path_id, path_system = context_from_path(workdir)
    name_cycle_match = re.search(r"runML_cycle_0*([12])", job_name, re.IGNORECASE)
    name_cycle = int(name_cycle_match.group(1)) if name_cycle_match else None
    name_id_match = re.search(r"([A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)_80", job_name)
    name_id = canonical_id(name_id_match.group(1)) if name_id_match else None
    name_system = system_from_text(job_name)
    issues = []
    if name_cycle is not None and path_cycle is not None and name_cycle != path_cycle:
        issues.append(f"cycle:name={name_cycle},path={path_cycle}")
    if name_id is not None and path_id is not None and name_id != path_id:
        issues.append(f"id:name={name_id},path={path_id}")
    if name_system is not None and path_system is not None and name_system != path_system:
        issues.append(f"system:name={name_system},path={path_system}")
    return ",".join(issues)


def slurm_current(skip: bool) -> tuple[dict[tuple[int, str, str, str], list[dict[str, str]]], str]:
    index: dict[tuple[int, str, str, str], list[dict[str, str]]] = defaultdict(list)
    if skip or shutil.which("squeue") is None:
        return index, "skipped or unavailable"
    user = os.environ.get("USER") or getpass.getuser()
    code, stdout, stderr = run_command(
        ["squeue", "-h", "-u", user, "-o", "%A|%T|%.250j|%M|%R|%Z"]
    )
    if code != 0:
        return index, f"squeue failed: {stderr.strip()}"
    for line in stdout.splitlines():
        parts = line.split("|", 5)
        if len(parts) != 6:
            continue
        job_id, state, name, elapsed, reason, workdir = parts
        cycle, peptoid_id, system, stage = job_context(name, workdir)
        if None in (cycle, peptoid_id, system, stage):
            continue
        index[(cycle, peptoid_id, system, stage)].append(
            {
                "job_id": job_id,
                "state": state,
                "name": name,
                "elapsed": elapsed,
                "reason": reason,
                "workdir": workdir,
                "name_path_mismatch": job_name_path_mismatch(name, workdir),
            }
        )
    return index, f"matched {sum(len(v) for v in index.values())} simulation jobs"


def slurm_history(
    skip: bool, start_date: str
) -> tuple[dict[tuple[int, str, str, str], list[dict[str, str]]], str]:
    index: dict[tuple[int, str, str, str], list[dict[str, str]]] = defaultdict(list)
    if skip or shutil.which("sacct") is None:
        return index, "skipped or unavailable"
    user = os.environ.get("USER") or getpass.getuser()
    fields = "JobIDRaw,JobName%250,State,ExitCode,Start,End,Elapsed,AllocCPUS,NodeList,WorkDir%300"
    code, stdout, stderr = run_command(
        ["sacct", "-u", user, "-S", start_date, "-X", "-n", "-P", "-o", fields],
        timeout=180,
    )
    if code != 0:
        return index, f"sacct failed: {stderr.strip()}"
    for line in stdout.splitlines():
        parts = line.split("|", 9)
        if len(parts) != 10:
            continue
        job_id, name, state, exit_code, start, end, elapsed, cpus, node, workdir = parts
        cycle, peptoid_id, system, stage = job_context(name, workdir)
        if None in (cycle, peptoid_id, system, stage):
            continue
        index[(cycle, peptoid_id, system, stage)].append(
            {
                "job_id": job_id,
                "state": state,
                "exit_code": exit_code,
                "start": start,
                "end": end,
                "elapsed": elapsed,
                "cpus": cpus,
                "node": node,
                "name": name,
                "workdir": workdir,
                "name_path_mismatch": job_name_path_mismatch(name, workdir),
            }
        )
    for jobs in index.values():
        jobs.sort(key=lambda item: (item["start"], item["job_id"]))
    return index, f"matched {sum(len(v) for v in index.values())} accounting rows since {start_date}"


def load_masterdata(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str | None], int]:
    rows, columns = read_csv(path)
    mapping = {
        "id": choose_column(columns, ["molecule_name", "peptoid_id", "id", "peptoid", "token_sequence"]),
        "system": choose_column(columns, ["system_key", "system", "membrane_system", "bilayer", "system_type"]),
        "analysis_complete": choose_column(columns, ["analysis_complete", "analysis_ready", "complete"]),
        "reached_100ns": choose_column(columns, ["reached_100ns"]),
        "timepoint_100ns": choose_column(columns, ["timepoint_100ns_found"]),
        "apl": choose_column(columns, ["APL_change", "DAPL", "delta_APL"]),
    }
    if not mapping["id"]:
        raise RuntimeError(f"Could not detect masterdata molecule ID column. Columns={columns}")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        peptoid_id = canonical_id(row.get(mapping["id"], ""))
        system = system_from_text(row.get(mapping["system"], ""), *row.values())
        if peptoid_id and system:
            grouped[(peptoid_id, system)].append(row)

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in grouped.items():
        valid_rows = []
        for row in group:
            checks = []
            if mapping["analysis_complete"]:
                checks.append(truth(row.get(mapping["analysis_complete"])) is True)
            if mapping["reached_100ns"]:
                checks.append(truth(row.get(mapping["reached_100ns"])) is True)
            if mapping["timepoint_100ns"]:
                checks.append(truth(row.get(mapping["timepoint_100ns"])) is True)
            if mapping["apl"]:
                checks.append(finite_float(row.get(mapping["apl"])) is not None)
            if checks and all(checks):
                valid_rows.append(row)
        result[key] = {"rows": len(group), "valid_rows": len(valid_rows), "valid": bool(valid_rows)}
    return result, mapping, len(rows)


def load_db_state(con: sqlite3.Connection) -> dict[str, Any]:
    tables = sqlite_tables(con)
    state: dict[str, Any] = {"tables": sorted(tables)}

    selections: dict[tuple[int, str], set[str]] = defaultdict(set)
    id_strategies: dict[tuple[int, str], set[str]] = defaultdict(set)
    selection_row_counts: Counter[tuple[int, str, str]] = Counter()
    if "selection_decisions" in tables:
        columns = sqlite_columns(con, "selection_decisions")
        mapping = {
            "id": choose_column(columns, ["id", "peptoid_id"]),
            "cycle": choose_column(columns, ["cycle"]),
            "strategy": choose_column(columns, ["strategy"]),
            "decision": choose_column(columns, ["decision"]),
            "archived": choose_column(columns, ["archived", "is_archived"]),
        }
        if all(mapping[name] for name in ("id", "cycle", "strategy", "decision")):
            for row in sqlite_all(con, "selection_decisions"):
                cycle = int_or_none(row.get(mapping["cycle"]))
                strategy = str(row.get(mapping["strategy"], "")).strip()
                decision = str(row.get(mapping["decision"], "")).strip().lower()
                if cycle not in (1, 2) or strategy not in CANONICAL_BY_CYCLE[cycle] or decision != "selected":
                    continue
                if mapping["archived"] and truth(row.get(mapping["archived"])) is True:
                    continue
                peptoid_id = canonical_id(row.get(mapping["id"], ""))
                if peptoid_id:
                    selections[(cycle, strategy)].add(peptoid_id)
                    id_strategies[(cycle, peptoid_id)].add(strategy)
                    selection_row_counts[(cycle, strategy, peptoid_id)] += 1
        state["selection_mapping"] = mapping

    observations: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "preferred": False})
    if "observations" in tables:
        columns = sqlite_columns(con, "observations")
        mapping = {
            "id": choose_column(columns, ["id", "peptoid_id"]),
            "preferred": choose_column(columns, ["preferred", "is_preferred"]),
            "ddapl": choose_column(columns, ["ddapl", "DDAPL", "delta_delta_apl"]),
        }
        if mapping["id"]:
            for row in sqlite_all(con, "observations"):
                peptoid_id = canonical_id(row.get(mapping["id"], ""))
                if not peptoid_id:
                    continue
                observations[peptoid_id]["rows"] += 1
                if mapping["preferred"] and truth(row.get(mapping["preferred"])) is True:
                    observations[peptoid_id]["preferred"] = True
                    if mapping["ddapl"]:
                        observations[peptoid_id]["ddapl"] = row.get(mapping["ddapl"])
        state["observation_mapping"] = mapping

    queue: dict[tuple[int, str], set[str]] = defaultdict(set)
    if "simulation_queue" in tables:
        columns = sqlite_columns(con, "simulation_queue")
        mapping = {
            "id": choose_column(columns, ["id", "peptoid_id"]),
            "cycle": choose_column(columns, ["cycle"]),
            "status": choose_column(columns, ["status", "state", "queue_status"]),
            "archived": choose_column(columns, ["archived", "is_archived"]),
        }
        if mapping["id"]:
            for row in sqlite_all(con, "simulation_queue"):
                if mapping["archived"] and truth(row.get(mapping["archived"])) is True:
                    continue
                cycle = int_or_none(row.get(mapping["cycle"])) if mapping["cycle"] else None
                peptoid_id = canonical_id(row.get(mapping["id"], ""))
                if cycle in (1, 2) and peptoid_id:
                    status = str(row.get(mapping["status"], "UNKNOWN")) if mapping["status"] else "UNKNOWN"
                    queue[(cycle, peptoid_id)].add(status)
        state["queue_mapping"] = mapping

    models: list[dict[str, Any]] = []
    if "models" in tables:
        columns = sqlite_columns(con, "models")
        mapping = {
            "name": choose_column(columns, ["model_name", "name", "model_id"]),
            "status": choose_column(columns, ["status", "state"]),
            "algorithm": choose_column(columns, ["algorithm", "winning_algorithm", "model_type"]),
            "representation": choose_column(columns, ["representation", "winning_representation"]),
            "training_run": choose_column(columns, ["training_run_id", "training_run"]),
            "prediction_run": choose_column(columns, ["prediction_run_id", "prediction_run"]),
        }
        for row in sqlite_all(con, "models"):
            models.append({key: row.get(column) if column else None for key, column in mapping.items()})
        state["model_mapping"] = mapping

    membership_counts: dict[str, Counter[str]] = defaultdict(Counter)
    if "dataset_membership" in tables:
        columns = sqlite_columns(con, "dataset_membership")
        model_col = choose_column(columns, ["model_name", "model", "model_id"])
        role_col = choose_column(columns, ["role", "membership_type", "split", "dataset_role"])
        if model_col:
            for row in sqlite_all(con, "dataset_membership"):
                model_name = str(row.get(model_col, ""))
                role = str(row.get(role_col, "member")) if role_col else "member"
                membership_counts[model_name][role] += 1

    state.update(
        selections=selections,
        selection_row_counts=selection_row_counts,
        id_strategies=id_strategies,
        observations=observations,
        queue=queue,
        models=models,
        membership_counts=membership_counts,
    )
    return state


def discover_all_live_physical_directories(
    root: Path,
) -> dict[tuple[str, str], list[Path]]:
    """Index live Boxes/2.1 directories without entering ML cleanup archives."""
    result: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for system in SYSTEMS:
        system_root = root / "Boxes" / "2.1" / system
        if not system_root.is_dir():
            continue
        for directory in system_root.rglob("*_80"):
            if not directory.is_dir():
                continue
            peptoid_id = canonical_id(directory.name)
            if peptoid_id:
                result[(peptoid_id, system)].append(directory)
    for paths in result.values():
        paths.sort()
    return result


def parse_slurm_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"unknown", "none", "n/a"}:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def overlapping_job_pairs(jobs: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    intervals = []
    for job in jobs:
        start = parse_slurm_datetime(job.get("start", ""))
        end = parse_slurm_datetime(job.get("end", ""))
        if start is None or end is None or end <= start:
            continue
        intervals.append((start, end, job.get("job_id", "")))
    overlaps: list[tuple[str, str]] = []
    for index, (start_a, end_a, id_a) in enumerate(intervals):
        for start_b, end_b, id_b in intervals[index + 1 :]:
            if start_a < end_b and start_b < end_a:
                overlaps.append((id_a, id_b))
    return overlaps


def directory_progress_label(directory: Path) -> str:
    labels = []
    for stage in STAGES:
        mdp_data = parse_mdp(directory / f"{stage}.mdp")
        log_data = parse_gromacs_log(directory / f"{stage}.log")
        target = mdp_data.get("nsteps") or log_data.get("log_nsteps")
        step = log_data.get("max_step")
        if target and step is not None and step >= target:
            state = "complete"
        elif step is not None and step > 0:
            state = f"step={step}"
        elif (directory / f"{stage}.tpr").is_file():
            state = "configured"
        else:
            state = "absent"
        labels.append(f"{stage}:{state}")
    return ",".join(labels)


def discover_physical_ids(root: Path) -> dict[tuple[int, str], set[str]]:
    result: dict[tuple[int, str], set[str]] = defaultdict(set)
    for cycle in (1, 2):
        for system in SYSTEMS:
            run_dir = root / "Boxes" / "2.1" / system / f"run_ML_cycle_{cycle:02d}"
            if not run_dir.is_dir():
                continue
            for directory in run_dir.iterdir():
                if directory.is_dir() and directory.name.endswith("_80"):
                    peptoid_id = canonical_id(directory.name)
                    if peptoid_id:
                        result[(cycle, peptoid_id)].add(system)
    return result


def stage_record(
    root: Path,
    cycle: int,
    peptoid_id: str,
    system: str,
    stage: str,
    canonical: bool,
    strategies: set[str],
    master: Mapping[tuple[str, str], Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    queue: Mapping[tuple[int, str], set[str]],
    current_jobs: Mapping[tuple[int, str, str, str], list[dict[str, str]]],
    history: Mapping[tuple[int, str, str, str], list[dict[str, str]]],
) -> dict[str, Any]:
    directory = root / "Boxes" / "2.1" / system / f"run_ML_cycle_{cycle:02d}" / f"{peptoid_id}_80"
    mdp = directory / f"{stage}.mdp"
    tpr = directory / f"{stage}.tpr"
    log = directory / f"{stage}.log"
    cpt = directory / f"{stage}.cpt"

    mdp_data = parse_mdp(mdp)
    log_data = parse_gromacs_log(log)
    dt_ps = mdp_data["dt_ps"] if mdp_data["dt_ps"] is not None else log_data["log_dt_ps"]
    nsteps = mdp_data["nsteps"] if mdp_data["nsteps"] is not None else log_data["log_nsteps"]
    planned_ns = nsteps * dt_ps / 1000 if nsteps is not None and dt_ps is not None else None
    max_step = log_data["max_step"]
    max_time_ps = log_data["max_time_ps"]
    if max_time_ps is None and max_step is not None and dt_ps is not None:
        max_time_ps = max_step * dt_ps
    completed_ns = max_time_ps / 1000 if max_time_ps is not None else None
    progress = 100 * max_step / nsteps if max_step is not None and nsteps and nsteps > 0 else None
    is_complete = bool(
        nsteps is not None
        and nsteps > 0
        and (
            (max_step is not None and max_step >= nsteps)
            or (max_time_ps is not None and dt_ps is not None and max_time_ps + dt_ps >= nsteps * dt_ps)
        )
    )
    cpt_exists = cpt.is_file() and cpt.stat().st_size > 0
    parameter_valid = dt_ps == 0.02 and nsteps == 5_000_000

    if not directory.is_dir():
        stage_state = "DIRECTORY_MISSING"
    elif is_complete:
        stage_state = "COMPLETE_100NS"
    elif max_step is not None and max_step > 0 and cpt_exists:
        stage_state = "INCOMPLETE_RESUMABLE"
    elif max_step is not None and max_step > 0:
        stage_state = "INCOMPLETE_NO_CHECKPOINT"
    elif tpr.is_file():
        stage_state = "CONFIGURED_NOT_STARTED"
    else:
        stage_state = "SETUP_REQUIRED"

    master_info = master.get((peptoid_id, system), {"rows": 0, "valid_rows": 0, "valid": False})
    observation = observations.get(peptoid_id, {"preferred": False})
    key = (cycle, peptoid_id, system, stage)
    active = current_jobs.get(key, [])
    old_jobs = history.get(key, [])
    recent_history = ";".join(
        f"{job['job_id']}:{job['state']}:{job['node']}" for job in old_jobs[-5:]
    )
    name_path_mismatches = ";".join(
        f"{job['job_id']}:{job['name_path_mismatch']}"
        for job in old_jobs
        if job.get("name_path_mismatch")
    )

    return {
        "cycle": cycle,
        "canonical": canonical,
        "strategies": "/".join(sorted(strategies)),
        "peptoid_id": peptoid_id,
        "system": system,
        "stage": stage,
        "directory": str(directory),
        "directory_exists": directory.is_dir(),
        "mdp_path": str(mdp),
        "mdp_exists": mdp.is_file(),
        "dt_ps": dt_ps,
        "nsteps": nsteps,
        "planned_ns": planned_ns,
        "parameter_valid": parameter_valid,
        "tpr_exists": tpr.is_file(),
        "log_exists": log.is_file(),
        "checkpoint_exists": cpt_exists,
        "checkpoint_size_bytes": cpt.stat().st_size if cpt_exists else 0,
        "max_step": max_step,
        "max_time_ps": max_time_ps,
        "completed_ns": completed_ns,
        "progress_percent": round(min(progress, 100.0), 4) if progress is not None else None,
        "finished_marker": log_data["finished_marker"],
        "fatal_marker": log_data["fatal_marker"],
        "last_performance_ns_day": log_data["last_performance_ns_day"],
        "last_checkpoint_step": log_data["last_checkpoint_step"],
        "stage_state": stage_state,
        "master_rows": master_info["rows"],
        "master_valid_rows": master_info["valid_rows"],
        "master_analysis_valid": master_info["valid"],
        "preferred_observation": bool(observation.get("preferred")),
        "db_queue_statuses": "/".join(sorted(queue.get((cycle, peptoid_id), set()))),
        "acquisition_state": "",
        "current_job_ids": "/".join(job["job_id"] for job in active),
        "current_job_states": "/".join(job["state"] for job in active),
        "recent_job_history": recent_history,
        "job_name_path_mismatches": name_path_mismatches,
        "recommended_action": "",
        "action_reason": "",
        "log_modified_utc": log_data["log_modified_utc"],
    }


def assign_actions(rows: list[dict[str, Any]]) -> dict[tuple[int, str], str]:
    by_acquisition: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_acquisition[(row["cycle"], row["peptoid_id"])].append(row)

    acquisition_states: dict[tuple[int, str], str] = {}
    for key, group in by_acquisition.items():
        canonical = any(row["canonical"] for row in group)
        preferred = any(row["preferred_observation"] for row in group)
        valid_systems = {
            row["system"] for row in group if row["master_analysis_valid"]
        }
        complete_keys = {
            (row["system"], row["stage"])
            for row in group
            if row["stage_state"] == "COMPLETE_100NS"
        }
        if not canonical:
            acquisition_state = "NONCANONICAL_PHYSICAL_PATH"
        elif preferred:
            acquisition_state = "SYNCED_REAL_OBSERVATION"
        elif valid_systems == set(SYSTEMS):
            acquisition_state = "VALID_MASTER_PAIR_READY_TO_SYNC"
        elif complete_keys == {(system, stage) for system in SYSTEMS for stage in STAGES}:
            acquisition_state = "PHYSICAL_COMPLETE_RUN_APL_ANALYSIS"
        else:
            acquisition_state = "PHYSICAL_INCOMPLETE"
        acquisition_states[key] = acquisition_state

        per_system = {system: {row["stage"]: row for row in group if row["system"] == system} for system in SYSTEMS}
        for row in group:
            row["acquisition_state"] = acquisition_state
            active = bool(row["current_job_ids"])

            if not canonical:
                row["recommended_action"] = "NONCANONICAL_REVIEW_DO_NOT_SUBMIT"
                row["action_reason"] = "Physical directory is not a current canonical Cycle-1/2 selection."
                continue

            if preferred:
                if active:
                    row["recommended_action"] = "CRITICAL_REAL_OBSERVATION_HAS_ACTIVE_JOB"
                    row["action_reason"] = "A preferred real observation already exists; active simulation may be duplicate."
                elif row["directory_exists"] and row["stage_state"] != "COMPLETE_100NS" and (row["tpr_exists"] or row["log_exists"]):
                    row["recommended_action"] = "REVIEW_REAL_OBSERVATION_PLUS_INCOMPLETE_CYCLE_PATH"
                    row["action_reason"] = "Preferred real observation already exists, but this cycle also contains an incomplete/configured physical stage; do not submit it automatically."
                else:
                    row["recommended_action"] = "NONE_ALREADY_SYNCED"
                    row["action_reason"] = "Preferred real observation exists in active_learning.db."
                continue

            if acquisition_state == "VALID_MASTER_PAIR_READY_TO_SYNC":
                if row["system"] == MAMMALIAN and row["stage"] == "Relax":
                    row["recommended_action"] = "SYNC_REVIEW_REQUIRED"
                    row["action_reason"] = "Both membrane rows are valid but preferred observation is absent."
                else:
                    row["recommended_action"] = "NONE_ANALYSED_PHYSICAL_RESULT_EXISTS"
                    row["action_reason"] = "Valid analysed pair exists; the single sync-review action is reported once for this peptoid."
                continue

            if row["master_analysis_valid"]:
                if active:
                    row["recommended_action"] = "CRITICAL_ANALYSED_SYSTEM_HAS_ACTIVE_JOB"
                    row["action_reason"] = "Valid 100 ns masterdata row exists; active simulation may be duplicate."
                else:
                    row["recommended_action"] = "NONE_ANALYSED_PHYSICAL_RESULT_EXISTS"
                    row["action_reason"] = "This membrane already has a valid analysed 100 ns result."
                continue

            pull = per_system[row["system"]].get("Pull")
            relax = per_system[row["system"]].get("Relax")
            pull_complete = pull is not None and pull["stage_state"] == "COMPLETE_100NS"

            if row["stage"] == "Relax" and not pull_complete:
                if active:
                    row["recommended_action"] = "CRITICAL_RELAX_ACTIVE_BEFORE_PULL_COMPLETE"
                    row["action_reason"] = "Pull is not proven complete for this membrane."
                else:
                    row["recommended_action"] = "BLOCKED_UNTIL_PULL_COMPLETE"
                    row["action_reason"] = "Pull must reach its configured 100 ns endpoint first."
                continue

            if active:
                if row["stage_state"] == "COMPLETE_100NS":
                    row["recommended_action"] = "CRITICAL_COMPLETE_STAGE_HAS_ACTIVE_JOB"
                    row["action_reason"] = "Stage is already complete but a matching Slurm job is active."
                else:
                    row["recommended_action"] = "ACTIVE_DO_NOT_RESUBMIT"
                    row["action_reason"] = "A matching running/pending Slurm job already exists."
                continue

            if not row["parameter_valid"] and row["mdp_exists"]:
                row["recommended_action"] = "MANUAL_PARAMETER_REVIEW"
                row["action_reason"] = f"Expected dt=0.02 and nsteps=5000000; found dt={row['dt_ps']} nsteps={row['nsteps']}."
            elif row["stage_state"] == "COMPLETE_100NS":
                if row["stage"] == "Relax" and pull_complete:
                    row["recommended_action"] = "RUN_APL_ANALYSIS_AFTER_PAIR_COMPLETE"
                    row["action_reason"] = "Pull and Relax are complete for this membrane; masterdata is not yet valid."
                else:
                    row["recommended_action"] = "NONE_STAGE_COMPLETE"
                    row["action_reason"] = "Configured stage endpoint reached."
            elif row["stage_state"] == "INCOMPLETE_RESUMABLE":
                row["recommended_action"] = f"RESUME_{row['stage'].upper()}_FROM_CHECKPOINT"
                row["action_reason"] = f"Incomplete at step {row['max_step']} with a non-empty checkpoint."
            elif row["stage_state"] == "INCOMPLETE_NO_CHECKPOINT":
                row["recommended_action"] = "MANUAL_RECOVERY_REVIEW"
                row["action_reason"] = "Simulation progressed but no non-empty checkpoint was found."
            elif row["stage_state"] == "CONFIGURED_NOT_STARTED":
                row["recommended_action"] = f"START_{row['stage'].upper()}"
                row["action_reason"] = "TPR exists but no simulation progress was detected."
            else:
                row["recommended_action"] = f"SETUP_{row['stage'].upper()}_REQUIRED"
                row["action_reason"] = "Required cycle directory/TPR/progress is absent."

    return acquisition_states


def fmt(value: Any, width: int = 42) -> str:
    if value is None or value == "":
        return "-"
    text = str(value).replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    rendered = [[fmt(value) for value in row] for row in rows]
    if not rendered:
        return ["(none)"]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    output = ["  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
    output.append("  ".join("-" * width for width in widths))
    output.extend("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in rendered)
    return output


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_parent = args.output_parent.resolve()
    master_path = root / "analysis" / "masterdataanalysis.csv"
    ml_root = root / "analysis" / "machinelearning"
    db_path = ml_root / "shared_data" / "active_learning" / "active_learning.db"
    workflow_path = ml_root / "position_resolved" / "scripts" / "workflow.py"
    config_path = ml_root / "config" / "model_search_config.json"

    required = [root, master_path, db_path, workflow_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Required paths missing: {missing}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_parent / f"audit_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "campaign_audit_report.txt"
    csv_path = output_dir / "campaign_stage_state.csv"

    master, master_mapping, master_count = load_masterdata(master_path)
    uri = f"file:{db_path.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only=ON")
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    try:
        db = load_db_state(con)
    finally:
        con.close()

    current_jobs, current_note = slurm_current(args.no_slurm)
    history, history_note = slurm_history(args.no_slurm, args.sacct_start)
    physical = discover_physical_ids(root)
    all_live_directories = discover_all_live_physical_directories(root)

    all_keys = set(physical) | set(db["id_strategies"])
    stage_rows: list[dict[str, Any]] = []
    for cycle, peptoid_id in sorted(all_keys):
        strategies = db["id_strategies"].get((cycle, peptoid_id), set())
        canonical = bool(strategies)
        for system in SYSTEMS:
            for stage in STAGES:
                stage_rows.append(
                    stage_record(
                        root, cycle, peptoid_id, system, stage, canonical,
                        strategies, master, db["observations"], db["queue"],
                        current_jobs, history,
                    )
                )

    acquisition_states = assign_actions(stage_rows)

    d1_ids = db["selections"].get((1, "D"), set())
    d2_ids = db["selections"].get((2, "D"), set())
    all_d_ids = d1_ids | d2_ids
    duplicate_d_selection_rows = {
        key: count
        for key, count in db["selection_row_counts"].items()
        if key[1] == "D" and count > 1
    }
    d_directory_duplicates = {
        key: paths
        for key, paths in all_live_directories.items()
        if key[0] in all_d_ids and len(paths) > 1
    }
    d_jobs_by_physical_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for (cycle, peptoid_id, system, stage), jobs in history.items():
        if peptoid_id in all_d_ids:
            d_jobs_by_physical_key[(peptoid_id, system, stage)].extend(jobs)
    d_job_audit = []
    for key, jobs in sorted(d_jobs_by_physical_key.items()):
        jobs.sort(key=lambda item: (item.get("start", ""), item.get("job_id", "")))
        overlaps = overlapping_job_pairs(jobs)
        distinct_workdirs = sorted({job.get("workdir", "") for job in jobs if job.get("workdir")})
        mismatches = [
            f"{job['job_id']}:{job['name_path_mismatch']}"
            for job in jobs
            if job.get("name_path_mismatch")
        ]
        if overlaps:
            classification = "OVERLAPPING_JOB_INTERVALS"
        elif len(jobs) > 1:
            classification = "MULTIPLE_SEQUENTIAL_JOB_IDS"
        else:
            classification = "ONE_JOB_ID"
        d_job_audit.append(
            {
                "key": key,
                "jobs": jobs,
                "overlaps": overlaps,
                "workdirs": distinct_workdirs,
                "mismatches": mismatches,
                "classification": classification,
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stage_rows)

    workflow_hash = sha256(workflow_path)
    source = workflow_path.read_text(encoding="utf-8", errors="replace")
    commands = sorted(set(re.findall(r"add_parser\(\s*['\"]([^'\"]+)", source)))
    config_summary = "missing"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config_summary = f"valid JSON; top-level keys={sorted(config)}"
        except Exception as exc:
            config_summary = f"INVALID JSON: {exc}"

    canonical_acquisitions = {
        key for key, strategies in db["id_strategies"].items() if strategies
    }
    acquisition_counts = Counter(acquisition_states[key] for key in canonical_acquisitions)
    action_counts = Counter(
        row["recommended_action"] for row in stage_rows if row["canonical"]
    )

    assertions = [
        ("SQLite integrity_check", integrity == "ok", integrity),
        ("workflow.py matches authoritative 2026-09-01 hash", workflow_hash == EXPECTED_WORKFLOW_SHA256, workflow_hash),
        ("Cycle-1 A-E canonical union = 38", len({pid for (cycle, pid), s in db["id_strategies"].items() if cycle == 1 and s}) == 38, len({pid for (cycle, pid), s in db["id_strategies"].items() if cycle == 1 and s})),
        ("Cycle-1 B has 15", len(db["selections"].get((1, "B"), set())) == 15, len(db["selections"].get((1, "B"), set()))),
        ("Corrected B1 contains Nf-Nfe-Nm", CORRECTED_B1 in db["selections"].get((1, "B"), set()), CORRECTED_B1 in db["selections"].get((1, "B"), set())),
        ("Corrected B1 excludes Nf-Nf-Nkeqm", INVALID_B1 not in db["selections"].get((1, "B"), set()), INVALID_B1 in db["selections"].get((1, "B"), set())),
        ("Cycle-2 A has 15", len(db["selections"].get((2, "A"), set())) == 15, len(db["selections"].get((2, "A"), set()))),
        ("Canonical Cycle-2 B is empty", len(db["selections"].get((2, "B"), set())) == 0, len(db["selections"].get((2, "B"), set()))),
        ("Cycle-2 D exact corrected set", db["selections"].get((2, "D"), set()) == EXPECTED_D2, f"missing={sorted(EXPECTED_D2-db['selections'].get((2, 'D'), set()))}; extra={sorted(db['selections'].get((2, 'D'), set())-EXPECTED_D2)}"),
        ("No duplicate active D selection rows", not duplicate_d_selection_rows, dict(duplicate_d_selection_rows)),
        ("No D1/D2 identity overlap", not (d1_ids & d2_ids), sorted(d1_ids & d2_ids)),
        ("No simultaneous overlapping D job intervals", not any(item["overlaps"] for item in d_job_audit), {str(item['key']): item['overlaps'] for item in d_job_audit if item['overlaps']}),
    ]

    lines: list[str] = []
    def section(title: str) -> None:
        lines.extend(["", "=" * 110, title, "=" * 110])

    lines.extend([
        "PFPEPTOIDS CONSOLIDATED CYCLE-1/CYCLE-2 CAMPAIGN AUDIT",
        f"Generated UTC: {now_utc().isoformat()}",
        "Safety: read-only project inspection; no simulation, APLAnalysis, sync, training, selection, or submission.",
        f"Project root: {root}",
        f"Detailed CSV: {csv_path}",
    ])

    section("1. SOURCE AND SCHEMA CHECKS")
    lines.extend([
        f"SQLite integrity_check: {integrity}",
        f"masterdata rows: {master_count}",
        f"masterdata detected columns: {master_mapping}",
        f"workflow.py SHA256: {workflow_hash}",
        f"workflow hash verdict: {'PASS' if workflow_hash == EXPECTED_WORKFLOW_SHA256 else 'WARNING_DIFFERENT'}",
        f"model_search_config.json: {config_summary}",
        f"current Slurm: {current_note}",
        f"Slurm history: {history_note}",
    ])

    section("2. CANONICAL LINEAGE ASSERTIONS")
    lines.extend(table(
        ["Assertion", "Result", "Observed/detail"],
        [(label, "PASS" if passed else "FAIL", detail) for label, passed, detail in assertions],
    ))

    section("3. CANONICAL SELECTION COUNTS")
    selection_rows = []
    for cycle in (1, 2):
        for strategy in sorted(CANONICAL_BY_CYCLE[cycle]):
            ids = db["selections"].get((cycle, strategy), set())
            selection_rows.append((cycle, strategy, len(ids)))
    lines.extend(table(["Cycle", "Strategy", "Selected unique IDs"], selection_rows))

    section("4. CANONICAL ACQUISITION STATE SUMMARY")
    lines.extend(table(
        ["Acquisition state", "Unique cycle/ID records"],
        sorted(acquisition_counts.items()),
    ))

    section("5. CANONICAL STAGE ACTION SUMMARY")
    lines.extend(table(["Recommended action", "Stage rows"], sorted(action_counts.items())))

    section("6. PER-ID CANONICAL STATE")
    per_id_rows = []
    for key in sorted(canonical_acquisitions):
        cycle, peptoid_id = key
        strategies = "/".join(sorted(db["id_strategies"][key]))
        gate = "B1_GATE" if peptoid_id == CORRECTED_B1 else ("CE_GATE" if peptoid_id in CORRECTED_CE else "-")
        actions = sorted({
            row["recommended_action"]
            for row in stage_rows
            if row["cycle"] == cycle and row["peptoid_id"] == peptoid_id
            and row["recommended_action"] not in {"NONE_STAGE_COMPLETE", "NONE_ALREADY_SYNCED", "NONE_ANALYSED_PHYSICAL_RESULT_EXISTS"}
        })
        per_id_rows.append((cycle, strategies, peptoid_id, gate, acquisition_states[key], ";".join(actions) or "NONE"))
    lines.extend(table(
        ["Cycle", "Strategies", "Peptoid", "Gate", "Acquisition state", "Outstanding actions"],
        per_id_rows,
    ))

    section("7. ACTIONABLE PHYSICAL STAGES")
    actionable = [
        row for row in stage_rows
        if row["canonical"]
        and row["recommended_action"] not in {
            "NONE_STAGE_COMPLETE", "NONE_ALREADY_SYNCED",
            "NONE_ANALYSED_PHYSICAL_RESULT_EXISTS", "BLOCKED_UNTIL_PULL_COMPLETE",
        }
    ]
    lines.extend(table(
        ["Cycle", "Strategies", "Peptoid", "System", "Stage", "Step", "%", "Action", "Current jobs"],
        [
            (
                row["cycle"], row["strategies"], row["peptoid_id"], row["system"],
                row["stage"], row["max_step"], row["progress_percent"],
                row["recommended_action"], row["current_job_ids"],
            )
            for row in actionable
        ],
    ))

    section("8. NONCANONICAL LIVE PHYSICAL DIRECTORIES")
    noncanonical_ids = sorted(key for key in physical if key not in canonical_acquisitions)
    lines.extend(table(
        ["Cycle", "Peptoid", "Systems with live directory"],
        [(cycle, peptoid_id, "/".join(sorted(physical[(cycle, peptoid_id)]))) for cycle, peptoid_id in noncanonical_ids],
    ))
    lines.append("These are review-only. Do not delete or submit from this list automatically.")

    section("9. D-LINEAGE DUPLICATION AUDIT")
    lines.extend([
        f"D1 unique IDs: {len(d1_ids)}",
        f"D2 unique IDs: {len(d2_ids)}",
        f"D1/D2 identity overlap: {sorted(d1_ids & d2_ids)}",
        f"Duplicate active D selection rows: {dict(duplicate_d_selection_rows)}",
        "",
        "Live-directory duplication by peptoid and membrane:",
    ])
    directory_duplicate_rows = []
    for (peptoid_id, system), paths in sorted(d_directory_duplicates.items()):
        details = "; ".join(
            f"{path} [{directory_progress_label(path)}]" for path in paths
        )
        directory_duplicate_rows.append((peptoid_id, system, len(paths), details))
    lines.extend(table(
        ["Peptoid", "System", "Live directories", "Paths and progress"],
        directory_duplicate_rows,
    ))
    lines.extend([
        "",
        "Slurm submission history by physical identity/system/stage:",
        "MULTIPLE_SEQUENTIAL_JOB_IDS is not by itself a duplicate simulation; it can represent cancellation/recovery from one checkpoint.",
        "OVERLAPPING_JOB_INTERVALS is strong evidence that two allocations ran the same physical stage simultaneously.",
    ])
    lines.extend(table(
        ["Peptoid", "System", "Stage", "Job IDs", "Classification", "Overlaps", "Distinct workdirs", "Name/path mismatch"],
        [
            (
                item["key"][0], item["key"][1], item["key"][2],
                "/".join(job["job_id"] for job in item["jobs"]),
                item["classification"],
                ";".join(f"{a}+{b}" for a, b in item["overlaps"]),
                len(item["workdirs"]),
                ";".join(item["mismatches"]),
            )
            for item in d_job_audit
        ],
    ))

    section("10. LOGICAL MODEL RECORDS")
    lines.extend(table(
        ["Model", "Status", "Representation", "Algorithm", "Training run", "Prediction run"],
        [
            (model.get("name"), model.get("status"), model.get("representation"),
             model.get("algorithm"), model.get("training_run"), model.get("prediction_run"))
            for model in db["models"]
        ],
    ))

    section("11. DATASET MEMBERSHIP COUNTS")
    membership_rows = []
    for model_name, counts in sorted(db["membership_counts"].items()):
        membership_rows.append((model_name, sum(counts.values()), json.dumps(dict(counts), sort_keys=True)))
    lines.extend(table(["Model", "Total rows", "Roles"], membership_rows))

    section("12. CONTROLLER CAPABILITY INVENTORY")
    lines.append(f"Argparse subcommands detected in live workflow.py ({len(commands)}):")
    lines.append("  " + ", ".join(commands) if commands else "  (none detected by static parser scan)")
    required_commands = ["validate", "status", "sync", "select-f", "select-g"]
    lines.extend(table(
        ["Required capability", "Detected"],
        [(command, "YES" if command in commands else "NO/REVIEW") for command in required_commands],
    ))
    lines.append("Static command detection is not proof of scientific correctness; live source review remains required before mutation.")

    section("13. DECISION GATES")
    lines.extend([
        "1. Do not submit from DB queue status or directory existence alone.",
        "2. Submit/resume only rows classified after reconciling canonical selection, valid masterdata, preferred observation, stage progress, checkpoint and active Slurm state.",
        "3. Cycle-1 B gate: finish/analyse/sync Nf-Nfe-Nm before corrected Model1B.",
        "4. Cycle-1 C/E gate: finish/analyse/sync the four corrected midpoint acquisitions before Model1C/Model1E and pooled Model1F.",
        "5. Model1G must not exist. Model2G is the first no-self-feedback G control.",
        "6. B2 remains empty until corrected Model1B is trained, registered and predicts the full 32,768 universe.",
        "7. Simulation submission remains a manual action outside workflow.py.",
    ])

    failed = [label for label, passed, _ in assertions if not passed]
    section("14. AUDIT VERDICT")
    if failed:
        lines.append("LINEAGE ASSERTION FAILURES: do not submit or mutate until explained.")
        lines.extend(f"  - {label}" for label in failed)
    else:
        lines.append("Canonical lineage assertions passed. Use actionable stage rows for review; no submissions were made.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("PFPEPTOIDS CAMPAIGN AUDIT COMPLETE")
    print(f"Report: {report_path}")
    print(f"CSV:    {csv_path}")
    print(f"Canonical acquisitions: {len(canonical_acquisitions)}")
    print(f"Physical directories represented: {sum(len(v) for v in physical.values())}")
    print(f"Actionable canonical stage rows: {len(actionable)}")
    print(f"Lineage assertion failures: {len(failed)}")
    print("No simulations, analyses, database rows, model files, selections, or Slurm jobs were changed.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Audit interrupted; no scientific project state was changed.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"AUDIT FAILED SAFELY: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("No scientific project state was changed.", file=sys.stderr)
        raise SystemExit(2)
