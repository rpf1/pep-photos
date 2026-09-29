#!/usr/bin/env python3
"""
verify_ne_nm_nfn_resume.py

Read-only verification of every Ne-Nm-Nfn physical copy in:

    run_ML_cycle_01
    run_ML_cycle_02

for:

    POPE75POPG25_W_WF
    POPC60CHOL40_W_WF

Checks:
- Pull/Relax MDP configuration
- Pull/Relax log progress
- Pull/Relax completion markers
- Pull/Relax checkpoints and timestamps
- current Slurm jobs
- historical Slurm accounting
- compares Cycle-1 and Cycle-2 copies
- determines which checkpoint is the appropriate resume candidate

NO FILES ARE CREATED, DELETED, MOVED, RENAMED OR MODIFIED.
NO JOBS ARE SUBMITTED.
"""

from __future__ import annotations

import getpass
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
)

BOX_ROOT = ROOT / "Boxes" / "2.1"

PEPTOID_ID = "Ne-Nm-Nfn"
PEPTOID_DIR = f"{PEPTOID_ID}_80"

BACTERIAL = "POPE75POPG25_W_WF"
MAMMALIAN = "POPC60CHOL40_W_WF"

SYSTEMS = (
    BACTERIAL,
    MAMMALIAN,
)

RUNS = (
    "run_ML_cycle_01",
    "run_ML_cycle_02",
)

STAGES = (
    "Pull",
    "Relax",
)

SACCT_START = "2026-08-01"


STEP_TIME_RE = re.compile(
    r"^[ \t]*Step[ \t]+Time[ \t]*\r?\n"
    r"[ \t]*([0-9]+)[ \t]+([-+0-9.Ee]+)",
    re.MULTILINE,
)


def run_command(command: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number != number:
        return None

    return number


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_mdp(path: Path) -> dict[str, Any]:
    result = {
        "dt_ps": None,
        "nsteps": None,
    }

    if not path.is_file():
        return result

    values: dict[str, str] = {}

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    for line in text.splitlines():
        line = line.split(";", 1)[0].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)

        values[key.strip().lower()] = value.strip()

    result["dt_ps"] = finite_float(values.get("dt"))
    result["nsteps"] = int_or_none(
        values.get("nsteps") or values.get("nstep")
    )

    return result


def parse_log(path: Path) -> dict[str, Any]:
    result = {
        "exists": False,
        "max_step": None,
        "max_time_ps": None,
        "last_checkpoint_step": None,
        "finished": False,
        "fatal": False,
        "performance_ns_day": None,
        "modified_utc": None,
        "log_dt_ps": None,
        "log_nsteps": None,
    }

    if not path.is_file():
        return result

    result["exists"] = True

    stat = path.stat()

    result["modified_utc"] = datetime.fromtimestamp(
        stat.st_mtime,
        timezone.utc,
    ).isoformat()

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    pairs: list[tuple[int, float]] = []

    for step_text, time_text in STEP_TIME_RE.findall(text):
        try:
            pairs.append(
                (
                    int(step_text),
                    float(time_text),
                )
            )
        except ValueError:
            pass

    if pairs:
        result["max_step"] = max(
            step for step, _ in pairs
        )

        result["max_time_ps"] = max(
            time_ps for _, time_ps in pairs
        )

    checkpoints = [
        int(value)
        for value in re.findall(
            r"Writing checkpoint, step\s+([0-9]+)",
            text,
        )
    ]

    if checkpoints:
        result["last_checkpoint_step"] = checkpoints[-1]

    performances = []

    for value in re.findall(
        r"Performance:\s+([-+0-9.Ee]+)",
        text,
    ):
        parsed = finite_float(value)

        if parsed is not None:
            performances.append(parsed)

    if performances:
        result["performance_ns_day"] = performances[-1]

    result["finished"] = bool(
        re.search(
            r"Finished mdrun|GROMACS reminds you",
            text,
            re.IGNORECASE,
        )
    )

    result["fatal"] = bool(
        re.search(
            r"Fatal error|"
            r"Inconsistency in user input|"
            r"Segmentation fault|"
            r"MPI_ABORT|"
            r"constraint error",
            text,
            re.IGNORECASE,
        )
    )

    dt_values = re.findall(
        r"^[ \t]*dt[ \t]*=[ \t]*([-+0-9.Ee]+)",
        text,
        re.MULTILINE,
    )

    nstep_values = re.findall(
        r"^[ \t]*nsteps[ \t]*=[ \t]*([0-9]+)",
        text,
        re.MULTILINE,
    )

    if dt_values:
        result["log_dt_ps"] = finite_float(dt_values[-1])

    if nstep_values:
        result["log_nsteps"] = int_or_none(nstep_values[-1])

    return result


def file_info(path: Path) -> str:
    if not path.exists():
        return "MISSING"

    stat = path.stat()

    modified = datetime.fromtimestamp(
        stat.st_mtime,
        timezone.utc,
    ).isoformat()

    return (
        f"exists=yes "
        f"size={stat.st_size:,} "
        f"modified={modified}"
    )


def inspect_stage(
    directory: Path,
    stage: str,
) -> dict[str, Any]:

    mdp = directory / f"{stage}.mdp"
    tpr = directory / f"{stage}.tpr"
    log = directory / f"{stage}.log"
    cpt = directory / f"{stage}.cpt"
    prev_cpt = directory / f"{stage}_prev.cpt"
    gro = directory / f"{stage}.gro"
    xtc = directory / f"{stage}.xtc"

    mdp_data = parse_mdp(mdp)
    log_data = parse_log(log)

    nsteps = (
        mdp_data["nsteps"]
        if mdp_data["nsteps"] is not None
        else log_data["log_nsteps"]
    )

    dt_ps = (
        mdp_data["dt_ps"]
        if mdp_data["dt_ps"] is not None
        else log_data["log_dt_ps"]
    )

    max_step = log_data["max_step"]

    progress = None

    if (
        max_step is not None
        and nsteps is not None
        and nsteps > 0
    ):
        progress = min(
            100.0,
            100.0 * max_step / nsteps,
        )

    complete = False

    if (
        max_step is not None
        and nsteps is not None
        and max_step >= nsteps
    ):
        complete = True

    elif log_data["finished"] and progress is not None:
        complete = progress >= 99.9

    resumable = (
        not complete
        and max_step is not None
        and max_step > 0
        and cpt.is_file()
        and cpt.stat().st_size > 0
    )

    return {
        "directory": directory,
        "stage": stage,
        "mdp": mdp,
        "tpr": tpr,
        "log": log,
        "cpt": cpt,
        "prev_cpt": prev_cpt,
        "gro": gro,
        "xtc": xtc,
        "dt_ps": dt_ps,
        "nsteps": nsteps,
        "max_step": max_step,
        "max_time_ps": log_data["max_time_ps"],
        "progress": progress,
        "complete": complete,
        "resumable": resumable,
        "finished": log_data["finished"],
        "fatal": log_data["fatal"],
        "last_checkpoint_step": log_data["last_checkpoint_step"],
        "performance_ns_day": log_data["performance_ns_day"],
        "log_modified_utc": log_data["modified_utc"],
    }


def print_stage(result: dict[str, Any]) -> None:
    print(f"    STAGE: {result['stage']}")

    print(
        f"        MDP:        {file_info(result['mdp'])}"
    )

    print(
        f"        TPR:        {file_info(result['tpr'])}"
    )

    print(
        f"        LOG:        {file_info(result['log'])}"
    )

    print(
        f"        CPT:        {file_info(result['cpt'])}"
    )

    print(
        f"        PREV CPT:   {file_info(result['prev_cpt'])}"
    )

    print(
        f"        GRO:        {file_info(result['gro'])}"
    )

    print(
        f"        XTC:        {file_info(result['xtc'])}"
    )

    print(
        f"        dt:         {result['dt_ps']} ps"
    )

    print(
        f"        nsteps:     {result['nsteps']}"
    )

    print(
        f"        max step:   {result['max_step']}"
    )

    print(
        f"        max time:   {result['max_time_ps']} ps"
    )

    if result["progress"] is None:
        progress_text = "-"
    else:
        progress_text = f"{result['progress']:.4f}%"

    print(
        f"        progress:   {progress_text}"
    )

    print(
        f"        complete:   {result['complete']}"
    )

    print(
        f"        resumable:  {result['resumable']}"
    )

    print(
        f"        finished:   {result['finished']}"
    )

    print(
        f"        fatal:      {result['fatal']}"
    )

    print(
        f"        last CPT step: "
        f"{result['last_checkpoint_step']}"
    )

    print(
        f"        performance: "
        f"{result['performance_ns_day']} ns/day"
    )

    print(
        f"        log modified UTC: "
        f"{result['log_modified_utc']}"
    )


def current_slurm_jobs() -> list[str]:
    if shutil.which("squeue") is None:
        return []

    user = getpass.getuser()

    code, stdout, _ = run_command(
        [
            "squeue",
            "-h",
            "-u",
            user,
            "-o",
            "%i|%j|%T|%R",
        ]
    )

    if code != 0:
        return []

    return [
        line
        for line in stdout.splitlines()
        if PEPTOID_ID.lower() in line.lower()
    ]


def historical_slurm_jobs() -> list[str]:
    if shutil.which("sacct") is None:
        return []

    code, stdout, _ = run_command(
        [
            "sacct",
            "-S",
            SACCT_START,
            "-X",
            "-n",
            "-P",
            "-o",
            (
                "JobIDRaw,JobName%180,State,"
                "ExitCode,Start,End,Elapsed"
            ),
        ],
        timeout=90,
    )

    if code != 0:
        return []

    return [
        line
        for line in stdout.splitlines()
        if PEPTOID_ID.lower() in line.lower()
    ]


def main() -> None:
    print("=" * 100)
    print("NE-NM-NFN READ-ONLY RESUME VERIFICATION")
    print("=" * 100)

    print(f"Project root: {ROOT}")
    print(f"Peptoid:      {PEPTOID_ID}")
    print()

    all_results: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for system in SYSTEMS:
        print("=" * 100)
        print(f"SYSTEM: {system}")
        print("=" * 100)

        for run_name in RUNS:
            directory = (
                BOX_ROOT
                / system
                / run_name
                / PEPTOID_DIR
            )

            print()
            print(f"RUN:       {run_name}")
            print(f"DIRECTORY: {directory}")
            print(f"EXISTS:    {directory.is_dir()}")
            print()

            for stage in STAGES:
                result = inspect_stage(
                    directory,
                    stage,
                )

                all_results[
                    (
                        system,
                        run_name,
                        stage,
                    )
                ] = result

                print_stage(result)
                print()

    print("=" * 100)
    print("CURRENT SLURM JOBS")
    print("=" * 100)

    current_jobs = current_slurm_jobs()

    if current_jobs:
        for line in current_jobs:
            print(line)
    else:
        print("No current Slurm jobs matched Ne-Nm-Nfn.")

    print()
    print("=" * 100)
    print("HISTORICAL SLURM JOBS")
    print("=" * 100)

    history = historical_slurm_jobs()

    if history:
        for line in history:
            print(line)
    else:
        print("No matching sacct rows found or sacct unavailable.")

    print()
    print("=" * 100)
    print("AUTOMATIC COMPARISON")
    print("=" * 100)

    c1_popc_pull = all_results[
        (
            MAMMALIAN,
            "run_ML_cycle_01",
            "Pull",
        )
    ]

    c2_popc_pull = all_results[
        (
            MAMMALIAN,
            "run_ML_cycle_02",
            "Pull",
        )
    ]

    c2_pope_pull = all_results[
        (
            BACTERIAL,
            "run_ML_cycle_02",
            "Pull",
        )
    ]

    c2_pope_relax = all_results[
        (
            BACTERIAL,
            "run_ML_cycle_02",
            "Relax",
        )
    ]

    print(
        "Cycle-1 POPC Pull progress: "
        f"{c1_popc_pull['progress']}"
    )

    print(
        "Cycle-2 POPC Pull progress: "
        f"{c2_popc_pull['progress']}"
    )

    print(
        "Cycle-2 POPE Pull complete: "
        f"{c2_pope_pull['complete']}"
    )

    print(
        "Cycle-2 POPE Relax complete: "
        f"{c2_pope_relax['complete']}"
    )

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)

    if current_jobs:
        print(
            "BLOCKED: a matching Ne-Nm-Nfn Slurm job is "
            "currently active. Do not submit another job."
        )
        return

    if not c2_popc_pull["directory"].is_dir():
        print(
            "BLOCKED: canonical Cycle-2 POPC directory is missing."
        )
        return

    if c2_popc_pull["complete"]:
        print(
            "Cycle-2 POPC Pull is already complete. "
            "Do not resume Pull."
        )

    elif c2_popc_pull["resumable"]:
        print(
            "PASS: Cycle-2 POPC Pull has an incomplete "
            "non-zero trajectory and a usable Pull.cpt."
        )

        if (
            c1_popc_pull["progress"] is not None
            and c2_popc_pull["progress"] is not None
            and c2_popc_pull["progress"]
            > c1_popc_pull["progress"]
        ):
            print(
                "PASS: Cycle-2 POPC Pull is further progressed "
                "than the accidental Cycle-1 copy."
            )

        print()
        print(
            "CANONICAL RESUME CANDIDATE:"
        )

        print(
            c2_popc_pull["directory"] / "Pull.cpt"
        )

    else:
        print(
            "BLOCKED: Cycle-2 POPC Pull is not complete and "
            "does not satisfy the checkpoint-resume test."
        )

    print()

    if (
        c2_pope_pull["complete"]
        and c2_pope_relax["complete"]
    ):
        print(
            "PASS: Cycle-2 POPE physical simulation is complete. "
            "Do not submit another POPE Pull or Relax job."
        )
    else:
        print(
            "WARNING: Cycle-2 POPE is not fully complete according "
            "to the live files. Review before proceeding."
        )

    print()
    print(
        "No files were modified and no jobs were submitted."
    )


if __name__ == "__main__":
    main()
