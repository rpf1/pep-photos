#!/usr/bin/env python3
"""
audit_and_quarantine_noncanonical_peptoid_runs.py

Audit PFPeptoids physical run-directory membership against active_learning.db.

Canonical interpretation:

    run_0
        = Model0 dataset membership

    run_ML_cycle_01
        = canonical Cycle-1 selected acquisitions

    run_ML_cycle_02
        = canonical Cycle-2 selected acquisitions

The script checks both:

    POPE75POPG25_W_WF
    POPC60CHOL40_W_WF

and identifies peptoid directories that are present in the wrong run folder.

Default mode:
    READ ONLY

Optional:
    --execute-unreferenced

will move ONLY noncanonical directories that:
    - are not referenced by masterdataanalysis.csv
    - are not referenced by active_learning.db observations
    - have no matching active Slurm job
    - do not collide with an existing quarantine destination

Destination layout:

    <bilayer>/accidental_duplicates/<original_run>/<peptoid_dir>

Example:

    POPC60CHOL40_W_WF/
        accidental_duplicates/
            run_ML_cycle_01/
                Ne-Nm-Nfn_80/

The original run-name layer is intentionally retained for provenance and
to prevent filename collisions.

This script NEVER deletes data.

Referenced physical observations are deliberately NOT moved automatically.
They require a provenance-aware migration.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
)

BOX_ROOT = ROOT / "Boxes" / "2.1"

DB_PATH = (
    ROOT
    / "analysis"
    / "machinelearning"
    / "shared_data"
    / "active_learning"
    / "active_learning.db"
)

MASTERDATA = ROOT / "analysis" / "masterdataanalysis.csv"

SYSTEMS = (
    "POPE75POPG25_W_WF",
    "POPC60CHOL40_W_WF",
)

RUN_NAMES = (
    "run_0",
    "run_ML_cycle_01",
    "run_ML_cycle_02",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--execute-unreferenced",
        action="store_true",
        help=(
            "Actually move only noncanonical directories that are "
            "unreferenced and have no matching active Slurm job."
        ),
    )

    return parser.parse_args()


def canonical_id(directory_name: str) -> str:
    text = directory_name.strip()

    if text.endswith("_80"):
        text = text[:-3]

    return text


def looks_like_peptoid(peptoid_id: str) -> bool:
    """
    Active-learning peptoids are three hyphen-separated monomer tokens.
    This intentionally ignores peptide directories such as FFPYGWKG_80.
    """
    return len(peptoid_id.split("-")) == 3


def connect_read_only() -> sqlite3.Connection:
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found:\n{DB_PATH}"
        )

    con = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
    )

    con.row_factory = sqlite3.Row

    con.execute(
        "PRAGMA query_only = ON"
    )

    integrity = con.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    if integrity != "ok":
        con.close()

        raise RuntimeError(
            f"SQLite integrity_check failed: {integrity}"
        )

    return con


def load_expected_membership(
    con: sqlite3.Connection,
) -> dict[str, set[str]]:

    expected: dict[str, set[str]] = {
        "run_0": set(),
        "run_ML_cycle_01": set(),
        "run_ML_cycle_02": set(),
    }

    model0_rows = con.execute(
        """
        SELECT DISTINCT id
        FROM dataset_membership
        WHERE model_id = 'Model0'
        """
    ).fetchall()

    expected["run_0"] = {
        str(row["id"])
        for row in model0_rows
    }

    for cycle, run_name in (
        (1, "run_ML_cycle_01"),
        (2, "run_ML_cycle_02"),
    ):
        rows = con.execute(
            """
            SELECT DISTINCT id
            FROM selection_decisions
            WHERE cycle = ?
              AND decision = 'selected'
            """,
            (cycle,),
        ).fetchall()

        expected[run_name] = {
            str(row["id"])
            for row in rows
        }

    return expected


def load_masterdata_references() -> set[str]:
    if not MASTERDATA.is_file():
        raise FileNotFoundError(
            f"masterdataanalysis.csv missing:\n{MASTERDATA}"
        )

    references: set[str] = set()

    with MASTERDATA.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(
                "masterdataanalysis.csv has no header."
            )

        if "system_dir" not in reader.fieldnames:
            raise RuntimeError(
                "masterdataanalysis.csv does not contain system_dir."
            )

        for row in reader:
            value = str(
                row.get("system_dir") or ""
            ).strip()

            if value:
                references.add(value.rstrip("/"))

    return references


def load_database_observation_references(
    con: sqlite3.Connection,
) -> set[str]:

    columns = {
        str(row["name"])
        for row in con.execute(
            "PRAGMA table_info(observations)"
        ).fetchall()
    }

    wanted = [
        column
        for column in (
            "bacteria_system_dir",
            "mammalian_system_dir",
        )
        if column in columns
    ]

    if not wanted:
        return set()

    sql = (
        "SELECT "
        + ", ".join(wanted)
        + " FROM observations"
    )

    references: set[str] = set()

    for row in con.execute(sql).fetchall():
        for column in wanted:
            value = str(
                row[column] or ""
            ).strip()

            if value:
                references.add(
                    value.rstrip("/")
                )

    return references


def current_slurm_jobs() -> list[dict[str, str]]:
    if shutil.which("squeue") is None:
        return []

    user = getpass.getuser()

    result = subprocess.run(
        [
            "squeue",
            "-h",
            "-u",
            user,
            "-o",
            "%i|%j|%T|%R",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        return []

    jobs: list[dict[str, str]] = []

    for line in result.stdout.splitlines():
        parts = line.split("|", 3)

        if len(parts) != 4:
            continue

        jobs.append(
            {
                "job_id": parts[0],
                "job_name": parts[1],
                "state": parts[2],
                "reason": parts[3],
            }
        )

    return jobs


def matching_active_jobs(
    peptoid_id: str,
    jobs: list[dict[str, str]],
) -> list[dict[str, str]]:

    needle1 = peptoid_id.lower()
    needle2 = f"{peptoid_id}_80".lower()

    matched = []

    for job in jobs:
        name = job["job_name"].lower()

        if (
            needle1 in name
            or needle2 in name
        ):
            matched.append(job)

    return matched


def directory_is_referenced(
    directory: Path,
    references: set[str],
) -> bool:

    target = str(directory).rstrip("/")

    if target in references:
        return True

    # Fail safely if a recorded path points at a file/subdirectory
    # beneath this physical system directory.
    prefix = target + "/"

    return any(
        reference.startswith(prefix)
        for reference in references
    )


def scan_run(
    system: str,
    run_name: str,
) -> list[Path]:

    root = BOX_ROOT / system / run_name

    if not root.is_dir():
        return []

    directories = []

    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue

        peptoid_id = canonical_id(path.name)

        if not looks_like_peptoid(peptoid_id):
            continue

        directories.append(path)

    return directories


def main() -> None:
    args = parse_args()

    print("=" * 110)
    print(
        "PFPEPTOIDS RUN DIRECTORY MEMBERSHIP / QUARANTINE AUDIT"
    )
    print("=" * 110)

    print(f"Project root: {ROOT}")
    print(f"Database:     {DB_PATH}")
    print(f"Masterdata:   {MASTERDATA}")
    print()

    if args.execute_unreferenced:
        print(
            "MODE: EXECUTE UNREFERENCED NONCANONICAL MOVES"
        )
    else:
        print(
            "MODE: READ ONLY"
        )

    print()

    with connect_read_only() as con:
        expected = load_expected_membership(con)

        master_refs = load_masterdata_references()

        db_refs = load_database_observation_references(
            con
        )

    all_refs = master_refs | db_refs

    jobs = current_slurm_jobs()

    print("=" * 110)
    print("CANONICAL EXPECTED COUNTS")
    print("=" * 110)

    for run_name in RUN_NAMES:
        print(
            f"{run_name:<20} "
            f"{len(expected[run_name]):>4} unique peptoids"
        )

    print()

    expected_runs_by_id: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for run_name, ids in expected.items():
        for peptoid_id in ids:
            expected_runs_by_id[peptoid_id].add(
                run_name
            )

    records: list[dict[str, Any]] = []

    for system in SYSTEMS:
        for run_name in RUN_NAMES:

            for directory in scan_run(
                system,
                run_name,
            ):
                peptoid_id = canonical_id(
                    directory.name
                )

                belongs_here = (
                    peptoid_id
                    in expected[run_name]
                )

                expected_elsewhere = sorted(
                    expected_runs_by_id.get(
                        peptoid_id,
                        set(),
                    )
                    - {run_name}
                )

                referenced_master = (
                    directory_is_referenced(
                        directory,
                        master_refs,
                    )
                )

                referenced_db = (
                    directory_is_referenced(
                        directory,
                        db_refs,
                    )
                )

                active_jobs = matching_active_jobs(
                    peptoid_id,
                    jobs,
                )

                destination = (
                    BOX_ROOT
                    / system
                    / "accidental_duplicates"
                    / run_name
                    / directory.name
                )

                if belongs_here:
                    classification = "KEEP_CANONICAL"
                    action = "KEEP"

                else:
                    if referenced_master or referenced_db:
                        classification = (
                            "NONCANONICAL_REFERENCED_PROVENANCE"
                        )
                        action = (
                            "BLOCK_MOVE_REQUIRES_PROVENANCE_MIGRATION"
                        )

                    elif active_jobs:
                        classification = (
                            "NONCANONICAL_ACTIVE_JOB"
                        )
                        action = (
                            "BLOCK_MOVE_ACTIVE_SLURM_JOB"
                        )

                    elif destination.exists():
                        classification = (
                            "NONCANONICAL_DESTINATION_EXISTS"
                        )
                        action = (
                            "BLOCK_MOVE_DESTINATION_EXISTS"
                        )

                    else:
                        classification = (
                            "NONCANONICAL_UNREFERENCED"
                        )
                        action = (
                            "SAFE_TO_QUARANTINE_UNREFERENCED"
                        )

                records.append(
                    {
                        "system": system,
                        "run_name": run_name,
                        "peptoid_id": peptoid_id,
                        "directory": directory,
                        "belongs_here": belongs_here,
                        "expected_elsewhere": expected_elsewhere,
                        "referenced_master": referenced_master,
                        "referenced_db": referenced_db,
                        "active_jobs": active_jobs,
                        "classification": classification,
                        "action": action,
                        "destination": destination,
                    }
                )

    print("=" * 110)
    print("NONCANONICAL PEPTOID DIRECTORIES")
    print("=" * 110)

    noncanonical = [
        row
        for row in records
        if not row["belongs_here"]
    ]

    if not noncanonical:
        print(
            "No noncanonical peptoid directories detected."
        )

    moved = 0
    blocked = 0
    safe = 0

    for row in noncanonical:

        print()
        print("-" * 110)

        print(
            f"Peptoid:       {row['peptoid_id']}"
        )

        print(
            f"Bilayer:       {row['system']}"
        )

        print(
            f"Current run:   {row['run_name']}"
        )

        if row["expected_elsewhere"]:
            print(
                "Expected run(s): "
                + ", ".join(
                    row["expected_elsewhere"]
                )
            )
        else:
            print(
                "Expected run(s): NONE in current canonical "
                "Model0/Cycle1/Cycle2 state"
            )

        print(
            f"Directory:     {row['directory']}"
        )

        print(
            f"Masterdata ref:{row['referenced_master']}"
        )

        print(
            f"Database ref:  {row['referenced_db']}"
        )

        if row["active_jobs"]:
            print("Active jobs:")

            for job in row["active_jobs"]:
                print(
                    f"    {job['job_id']} "
                    f"{job['state']} "
                    f"{job['job_name']}"
                )
        else:
            print("Active jobs:   none")

        print(
            f"Classification:{row['classification']}"
        )

        print(
            f"Action:        {row['action']}"
        )

        print(
            f"Destination:   {row['destination']}"
        )

        if (
            row["action"]
            == "SAFE_TO_QUARANTINE_UNREFERENCED"
        ):
            safe += 1

            if args.execute_unreferenced:

                row["destination"].parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.move(
                    str(row["directory"]),
                    str(row["destination"]),
                )

                moved += 1

                print(
                    "RESULT:        MOVED"
                )

            else:
                print(
                    "RESULT:        DRY RUN ONLY"
                )

        else:
            blocked += 1

            print(
                "RESULT:        NOT MOVED"
            )

    print()
    print("=" * 110)
    print("CANONICAL DIRECTORY CHECK")
    print("=" * 110)

    for system in SYSTEMS:
        print()
        print(system)

        for run_name in RUN_NAMES:

            present = {
                canonical_id(path.name)
                for path in scan_run(
                    system,
                    run_name,
                )
            }

            expected_here = expected[run_name]

            correct = present & expected_here
            unexpected = present - expected_here
            absent = expected_here - present

            print()
            print(f"  {run_name}")
            print(
                f"    canonical dirs present: "
                f"{len(correct)}"
            )
            print(
                f"    noncanonical dirs present: "
                f"{len(unexpected)}"
            )
            print(
                f"    canonical IDs without a directory here: "
                f"{len(absent)}"
            )

            if unexpected:
                print(
                    "    unexpected:"
                )

                for peptoid_id in sorted(
                    unexpected
                ):
                    print(
                        f"        {peptoid_id}"
                    )

            # Missing physical directories are NOT automatically errors.
            # A selection may legitimately reuse physical data generated
            # during an earlier cycle/branch.
            if absent:
                print(
                    "    no directory in this run "
                    "(may legitimately reuse older physical data):"
                )

                for peptoid_id in sorted(
                    absent
                ):
                    print(
                        f"        {peptoid_id}"
                    )

    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    print(
        f"Noncanonical directories:       "
        f"{len(noncanonical)}"
    )

    print(
        f"Safe unreferenced quarantine:   "
        f"{safe}"
    )

    print(
        f"Blocked/requires review:         "
        f"{blocked}"
    )

    if args.execute_unreferenced:
        print(
            f"Directories actually moved:      "
            f"{moved}"
        )
    else:
        print(
            "Directories actually moved:      0"
        )

    print()

    if not args.execute_unreferenced:
        print(
            "READ-ONLY audit complete."
        )
        print(
            "Do NOT run --execute-unreferenced until "
            "the reported move set has been reviewed."
        )
    else:
        print(
            "Execution complete."
        )
        print(
            "Only unreferenced noncanonical directories "
            "were eligible to move."
        )

    print()
    print(
        "No directories classified as referenced provenance "
        "were moved automatically."
    )


if __name__ == "__main__":
    main()
