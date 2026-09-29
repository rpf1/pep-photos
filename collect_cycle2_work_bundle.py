#!/usr/bin/env python3

from __future__ import annotations

import getpass
import hashlib
import json
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
)

ANALYSIS = ROOT / "analysis"

DB = (
    ANALYSIS
    / "machinelearning"
    / "shared_data"
    / "active_learning"
    / "active_learning.db"
)

MASTERDATA = ANALYSIS / "masterdataanalysis.csv"

WORKFLOW = (
    ANALYSIS
    / "machinelearning"
    / "position_resolved"
    / "scripts"
    / "workflow.py"
)

SCRIPT_INDEX = ROOT / "SCRIPT_INDEX.md"


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT = (
    ANALYSIS
    / "work_handoff_bundles"
    / f"cycle2_recovery_{TIMESTAMP}"
)

FILES = OUTPUT / "files"
LIVE = OUTPUT / "live_evidence"
CANDIDATES = OUTPUT / "candidate_stage_files"


CANDIDATE_STAGES = [
    (
        "Ne-Nm-Nfn",
        "POPC60CHOL40_W_WF",
        "Pull",
    ),

    (
        "Nfe-Nfnap-Nl",
        "POPC60CHOL40_W_WF",
        "Relax",
    ),

    (
        "Nfex-Nfnap-Nk",
        "POPC60CHOL40_W_WF",
        "Relax",
    ),

    (
        "Ni-Nfnap-Nl",
        "POPE75POPG25_W_WF",
        "Relax",
    ),

    (
        "Ni-Nfnap-Nl",
        "POPC60CHOL40_W_WF",
        "Relax",
    ),

    (
        "Nk-Nfex-Nfnap",
        "POPE75POPG25_W_WF",
        "Relax",
    ),

    (
        "Nk-Nfex-Nfnap",
        "POPC60CHOL40_W_WF",
        "Relax",
    ),

    (
        "Nk-Nfnap-Nl",
        "POPE75POPG25_W_WF",
        "Relax",
    ),

    (
        "Nk-Nv-Nwe",
        "POPC60CHOL40_W_WF",
        "Relax",
    ),

    (
        "Nl-Nfnap-Nfe",
        "POPE75POPG25_W_WF",
        "Relax",
    ),

    (
        "Nl-Nfnap-Nfe",
        "POPC60CHOL40_W_WF",
        "Relax",
    ),

    (
        "Nl-Nfnap-Nk",
        "POPE75POPG25_W_WF",
        "Relax",
    ),

    (
        "Nl-Nfnap-Nk",
        "POPC60CHOL40_W_WF",
        "Relax",
    ),

    (
        "Nv-Nfnap-Nk",
        "POPE75POPG25_W_WF",
        "Relax",
    ),

    (
        "Nv-Nfnap-Nk",
        "POPC60CHOL40_W_WF",
        "Relax",
    ),
]


SEARCH_NAMES = {
    "audit_pfpeptoids_campaign_state.py",
    "audit_and_quarantine_noncanonical_peptoid_runs.py",
    "verify_ne_nm_nfn_resume.py",
    "Simulations.py",
    "simulations.py",
    "aplanalysis.py",
    "APLanalysis.py",
    "PFPeptoidsAnalysisSuite.py",
    "Pull_recovery_mpirun_test.sh",
    "create_corrected_d2_simulations.py",
    "create_d2_simulations.py",
    "submit_d2_simulations.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def copy_with_structure(path: Path) -> Path:
    relative = path.relative_to(ROOT)

    destination = FILES / relative

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        path,
        destination,
    )

    return destination


def run_command(
    command: list[str],
    output_path: Path,
) -> None:

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )

    output_path.write_text(
        "COMMAND:\n"
        + " ".join(command)
        + "\n\nRETURN CODE:\n"
        + str(result.returncode)
        + "\n\nSTDOUT:\n"
        + result.stdout
        + "\n\nSTDERR:\n"
        + result.stderr,
        encoding="utf-8",
    )


def create_sqlite_backup() -> Path:
    destination = (
        FILES
        / "analysis"
        / "machinelearning"
        / "shared_data"
        / "active_learning"
        / "active_learning_snapshot.db"
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    target = sqlite3.connect(destination)

    try:
        source.backup(target)
    finally:
        target.close()
        source.close()

    return destination


def copy_candidate_files() -> list[dict]:
    manifest = []

    allowed_names = {
        "Pull.mdp",
        "Pull.tpr",
        "Pull.log",
        "Pull.cpt",
        "Pull_prev.cpt",
        "Pull.gro",

        "Relax.mdp",
        "Relax.tpr",
        "Relax.log",
        "Relax.cpt",
        "Relax_prev.cpt",
        "Relax.gro",

        "topol.top",
        "index.ndx",
    }

    for peptoid, system, stage in CANDIDATE_STAGES:

        physical = (
            ROOT
            / "Boxes"
            / "2.1"
            / system
            / "run_ML_cycle_02"
            / f"{peptoid}_80"
        )

        record = {
            "peptoid": peptoid,
            "system": system,
            "stage": stage,
            "physical_directory": str(physical),
            "directory_exists": physical.is_dir(),
            "files": [],
        }

        destination_root = (
            CANDIDATES
            / system
            / "run_ML_cycle_02"
            / f"{peptoid}_80"
        )

        if physical.is_dir():

            for name in sorted(allowed_names):
                source = physical / name

                if not source.is_file():
                    continue

                stat = source.stat()

                file_record = {
                    "name": name,
                    "path": str(source),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "sha256": sha256(source),
                }

                record["files"].append(
                    file_record
                )

                # Avoid copying huge scientific files.
                # These selected restart/setup files should normally
                # remain well below this threshold.
                if stat.st_size <= 10 * 1024 * 1024:

                    destination_root.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    shutil.copy2(
                        source,
                        destination_root / name,
                    )

        manifest.append(record)

    return manifest


def main() -> None:

    OUTPUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    FILES.mkdir(parents=True)
    LIVE.mkdir(parents=True)
    CANDIDATES.mkdir(parents=True)

    copied = []

    print(f"Output directory:\n{OUTPUT}")
    print()

    # ------------------------------------------------------------------
    # Core authoritative files
    # ------------------------------------------------------------------

    if not DB.is_file():
        raise FileNotFoundError(DB)

    db_snapshot = create_sqlite_backup()

    copied.append(db_snapshot)

    for path in (
        MASTERDATA,
        WORKFLOW,
        SCRIPT_INDEX,
    ):
        if path.is_file():
            copied.append(
                copy_with_structure(path)
            )

    # ------------------------------------------------------------------
    # Locate relevant project scripts
    # ------------------------------------------------------------------

    found_scripts = []

    for path in ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.name not in SEARCH_NAMES:
            continue

        found_scripts.append(path)

        try:
            copied.append(
                copy_with_structure(path)
            )
        except Exception as exc:
            print(
                f"WARNING copying {path}: {exc}"
            )

    # ------------------------------------------------------------------
    # Copy latest campaign audit outputs
    # ------------------------------------------------------------------

    audit_root = (
        ANALYSIS
        / "campaign_state_audits"
    )

    latest_audit = None

    if audit_root.is_dir():

        audit_dirs = sorted(
            [
                path
                for path in audit_root.iterdir()
                if path.is_dir()
            ],
            key=lambda path: path.name,
        )

        if audit_dirs:
            latest_audit = audit_dirs[-1]

            destination = (
                FILES
                / "analysis"
                / "campaign_state_audits"
                / latest_audit.name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True,
            )

            for source in latest_audit.iterdir():

                if not source.is_file():
                    continue

                if source.suffix.lower() not in {
                    ".txt",
                    ".csv",
                    ".json",
                    ".md",
                }:
                    continue

                shutil.copy2(
                    source,
                    destination / source.name,
                )

    # ------------------------------------------------------------------
    # Fresh live Slurm evidence
    # ------------------------------------------------------------------

    user = getpass.getuser()

    run_command(
        [
            "squeue",
            "-u",
            user,
            "-o",
            "%i|%j|%T|%M|%l|%R",
        ],
        LIVE / "squeue_current.txt",
    )

    if shutil.which("sacct"):

        run_command(
            [
                "sacct",
                "-S",
                "2026-08-01",
                "-X",
                "-n",
                "-P",
                "-o",
                (
                    "JobIDRaw,JobName%180,State,"
                    "ExitCode,Start,End,Elapsed"
                ),
            ],
            LIVE / "sacct_since_20260801.txt",
        )

    # ------------------------------------------------------------------
    # Fresh read-only project audit outputs
    # ------------------------------------------------------------------

    membership_script = (
        ANALYSIS
        / "audit_and_quarantine_noncanonical_peptoid_runs.py"
    )

    if membership_script.is_file():

        run_command(
            [
                "python",
                str(membership_script),
            ],
            LIVE / "membership_audit.txt",
        )

    verify_script = (
        ANALYSIS
        / "verify_ne_nm_nfn_resume.py"
    )

    if verify_script.is_file():

        run_command(
            [
                "python",
                str(verify_script),
            ],
            LIVE / "verify_ne_nm_nfn_resume.txt",
        )

    # ------------------------------------------------------------------
    # Candidate-stage files and hashes
    # ------------------------------------------------------------------

    candidate_manifest = copy_candidate_files()

    (
        LIVE / "candidate_stage_manifest.json"
    ).write_text(
        json.dumps(
            candidate_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # General manifest
    # ------------------------------------------------------------------

    manifest = {
        "generated": datetime.now().isoformat(),
        "project_root": str(ROOT),
        "output": str(OUTPUT),
        "database_source": str(DB),
        "database_snapshot": str(db_snapshot),
        "masterdata": str(MASTERDATA),
        "workflow": str(WORKFLOW),
        "script_index": str(SCRIPT_INDEX),
        "latest_audit": (
            str(latest_audit)
            if latest_audit
            else None
        ),
        "found_scripts": [
            str(path)
            for path in sorted(found_scripts)
        ],
        "copied_files": [
            str(path)
            for path in copied
        ],
    }

    (
        OUTPUT / "bundle_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Create compressed archive
    # ------------------------------------------------------------------

    archive = shutil.make_archive(
        str(OUTPUT),
        "gztar",
        root_dir=OUTPUT.parent,
        base_dir=OUTPUT.name,
    )

    print("=" * 80)
    print("WORK BUNDLE COMPLETE")
    print("=" * 80)

    print(f"Directory:\n{OUTPUT}")
    print()

    print(f"Archive:\n{archive}")
    print()

    print(
        "Source campaign files were not modified."
    )

    print(
        "No simulation, APL, ML, sync, selection, "
        "move or Slurm submission was performed."
    )


if __name__ == "__main__":
    raise SystemExit(
        "RETIRED: collect_cycle2_work_bundle.py caused recursive destination "
        "re-ingestion in the 2026-09-09 Cycle-2 recovery bundle. "
        "Do not rerun this historical collector. Use a reviewed external-staging "
        "collector for any future handoff."
    )
