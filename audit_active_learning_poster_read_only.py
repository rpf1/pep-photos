#!/usr/bin/env python3
"""Read-only PFPeptoids audit for choosing an active-learning poster figure.

This script deliberately does not run APLAnalysis, workflow sync, training,
prediction, selection, or Slurm submission.  It reuses the existing consolidated
campaign audit, then adds:

* one-row-per-acquisition full/partial cohort classification;
* a stage-level list of physical work that may still be required;
* an inventory of existing ML figures, dimensions, vector/raster format, and
  likely source scripts;
* an inventory of Python scripts that generate scientific figures.

All new files are written beneath one timestamped directory under
analysis/poster_active_learning_audits/.
"""

from __future__ import annotations

import ast
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids")
ANALYSIS = ROOT / "analysis"
ML = ANALYSIS / "machinelearning"
EXPECTED_SYSTEMS = {"POPE75POPG25_W_WF", "POPC60CHOL40_W_WF"}
EXPECTED_STAGES = {"Pull", "Relax"}
FIGURE_SUFFIXES = {".png", ".pdf", ".svg", ".tif", ".tiff", ".eps"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def natural_key(value: object) -> tuple:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value)))


def find_campaign_audit() -> Path:
    marker = "Consolidated read-only PFPeptoids Cycle-1/Cycle-2 campaign audit."
    matches: list[Path] = []
    this_script = Path(__file__).resolve()
    for path in ANALYSIS.rglob("*.py"):
        if path.resolve() == this_script:
            continue
        try:
            if marker in path.read_text(encoding="utf-8", errors="replace")[:12000]:
                matches.append(path.resolve())
        except OSError:
            continue
    matches = sorted(set(matches))
    if len(matches) != 1:
        detail = "\n".join(f"  {p}" for p in matches) or "  (none)"
        raise RuntimeError(
            "Expected exactly one existing consolidated campaign-audit script, "
            f"but found {len(matches)}:\n{detail}"
        )
    return matches[0]


def run_campaign_audit(script: Path, output_parent: Path) -> tuple[int, str, Path, Path]:
    command = [
        sys.executable,
        str(script),
        "--root",
        str(ROOT),
        "--output-parent",
        str(output_parent),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    console = (result.stdout or "") + ("\nSTDERR\n" + result.stderr if result.stderr else "")
    report_match = re.search(r"^Report:\s+(.+)$", result.stdout or "", re.MULTILINE)
    csv_match = re.search(r"^CSV:\s+(.+)$", result.stdout or "", re.MULTILINE)
    if not report_match or not csv_match:
        raise RuntimeError(
            "The consolidated campaign audit did not report its output paths.\n"
            f"Exit code: {result.returncode}\n{console[-5000:]}"
        )
    report = Path(report_match.group(1).strip())
    detailed = Path(csv_match.group(1).strip())
    if not report.is_file() or not detailed.is_file():
        raise RuntimeError(f"Campaign-audit outputs are missing: {report}; {detailed}")
    return result.returncode, console, report, detailed


def aggregate_campaign(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if truth(row.get("canonical")):
            grouped[(int(row["cycle"]), row["peptoid_id"])].append(row)

    cohorts: list[dict[str, object]] = []
    pending: list[dict[str, str]] = []
    for (cycle, peptoid), group in sorted(grouped.items(), key=lambda item: (item[0][0], natural_key(item[0][1]))):
        strategies = sorted({x for row in group for x in row.get("strategies", "").split("/") if x})
        valid_systems = {
            row["system"] for row in group
            if row.get("system") in EXPECTED_SYSTEMS and truth(row.get("master_analysis_valid"))
        }
        preferred = any(truth(row.get("preferred_observation")) for row in group)
        complete_stage_keys = {
            (row.get("system", ""), row.get("stage", ""))
            for row in group
            if row.get("stage_state") == "COMPLETE_100NS"
        }
        required_stage_keys = {(system, stage) for system in EXPECTED_SYSTEMS for stage in EXPECTED_STAGES}
        missing_stage_keys = sorted(required_stage_keys - complete_stage_keys)
        full_measured = valid_systems == EXPECTED_SYSTEMS
        if full_measured and preferred:
            classification = "FULL_MEASURED_AND_SYNCED"
        elif full_measured:
            classification = "FULL_MEASURED_MASTERDATA_NOT_SYNCED"
        elif valid_systems:
            classification = "PARTIAL_MEASURED_ONE_SYSTEM"
        else:
            classification = "NO_VALID_MEASURED_PAIR"

        actions = sorted({
            row.get("recommended_action", "") for row in group
            if row.get("recommended_action", "") and not row.get("recommended_action", "").startswith("NONE_")
        })
        cohorts.append({
            "cycle": cycle,
            "peptoid_id": peptoid,
            "strategies": "/".join(strategies),
            "classification": classification,
            "valid_analysed_systems": "/".join(sorted(valid_systems)),
            "preferred_observation_synced": preferred,
            "complete_physical_stages_of_4": len(complete_stage_keys & required_stage_keys),
            "missing_physical_stages": "; ".join(f"{s}:{p}" for s, p in missing_stage_keys),
            "recommended_actions": "; ".join(actions),
        })

        if not full_measured:
            for row in group:
                action = row.get("recommended_action", "")
                if action and not action.startswith("NONE_"):
                    if action.startswith(("RESUME_", "SETUP_", "START_", "SUBMIT_")):
                        action_class = "SIMULATION_REVIEW"
                    elif action.startswith(("RUN_APL_", "SYNC_")):
                        action_class = "ANALYSIS_OR_SYNC"
                    elif action.startswith("BLOCKED_"):
                        action_class = "WAIT_FOR_PREREQUISITE"
                    else:
                        action_class = "MANUAL_REVIEW"
                    pending.append({
                        "cycle": row.get("cycle", ""),
                        "strategies": row.get("strategies", ""),
                        "peptoid_id": row.get("peptoid_id", ""),
                        "system": row.get("system", ""),
                        "stage": row.get("stage", ""),
                        "stage_state": row.get("stage_state", ""),
                        "completed_ns": row.get("completed_ns", ""),
                        "checkpoint_exists": row.get("checkpoint_exists", ""),
                        "current_job_ids": row.get("current_job_ids", ""),
                        "action_class": action_class,
                        "recommended_action": action,
                        "action_reason": row.get("action_reason", ""),
                        "directory": row.get("directory", ""),
                    })
    return cohorts, pending


def script_metadata(path: Path) -> dict[str, object] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not any(token in text for token in ("savefig", "matplotlib", "seaborn", "plotly")):
        return None
    functions: list[str] = []
    try:
        tree = ast.parse(text, filename=str(path))
        functions = [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(word in node.name.lower() for word in ("plot", "figure", "heatmap", "scatter", "distribution", "rank"))
        ]
    except SyntaxError:
        pass
    output_literals = sorted(set(re.findall(
        r"[\"']([^\"']+\.(?:png|pdf|svg|tiff?|eps))[\"']", text, flags=re.IGNORECASE
    )))
    return {
        "script_path": str(path.resolve()),
        "savefig_mentions": text.count("savefig"),
        "plot_functions": "; ".join(sorted(set(functions))),
        "literal_output_names": "; ".join(output_literals[:80]),
    }


def figure_score(path: Path) -> int:
    name = str(path).lower()
    positive = {
        "predicted_vs_actual": 35,
        "actual_vs_predicted": 35,
        "prediction_outcome": 32,
        "prospective": 28,
        "acquisition": 24,
        "selection": 22,
        "full_space": 20,
        "32768": 18,
        "distribution": 16,
        "ranked": 14,
        "residual": 14,
        "model_comparison": 10,
        "measured_vs_predicted": 10,
        "benchmark": 4,
    }
    negative = {
        "snapshot": -40,
        "water": -30,
        "apl_vs_time": -24,
        "binned_apl": -20,
        "trajectory": -15,
    }
    return sum(weight for token, weight in {**positive, **negative}.items() if token in name)


def image_metadata(path: Path) -> tuple[str, str, str, str]:
    if path.suffix.lower() in {".pdf", ".svg", ".eps"}:
        return "vector", "", "", ""
    try:
        from PIL import Image
        with Image.open(path) as image:
            dpi = image.info.get("dpi", "")
            if isinstance(dpi, tuple):
                dpi = "x".join(f"{float(x):.1f}" for x in dpi)
            return "raster", str(image.width), str(image.height), str(dpi)
    except Exception as exc:
        return "raster-unreadable", "", "", type(exc).__name__


def inventory_figures(script_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    script_text: dict[str, str] = {}
    for row in script_rows:
        path = Path(str(row["script_path"]))
        try:
            script_text[str(path)] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass

    figures: list[dict[str, object]] = []
    skip_dirs = {".git", "__pycache__", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(ML):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in FIGURE_SUFFIXES:
                continue
            kind, width, height, dpi = image_metadata(path)
            sources = [script for script, text in script_text.items() if filename in text]
            stat = path.stat()
            figures.append({
                "poster_relevance_score": figure_score(path),
                "format_class": kind,
                "width_px": width,
                "height_px": height,
                "dpi_metadata": dpi,
                "size_mb": f"{stat.st_size / 1024 / 1024:.3f}",
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "likely_source_scripts": "; ".join(sources),
                "figure_path": str(path.resolve()),
            })
    return sorted(
        figures,
        key=lambda row: (-int(row["poster_relevance_score"]), str(row["modified_utc"]), str(row["figure_path"])),
        reverse=False,
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not ROOT.is_dir() or not ML.is_dir():
        raise RuntimeError(f"Project or ML directory not found beneath {ROOT}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ANALYSIS / "poster_active_learning_audits" / f"audit_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    campaign_script = find_campaign_audit()
    campaign_code, campaign_console, campaign_report, campaign_csv = run_campaign_audit(
        campaign_script, output_dir / "campaign_state"
    )
    (output_dir / "campaign_audit_console.txt").write_text(campaign_console, encoding="utf-8")

    with campaign_csv.open(newline="", encoding="utf-8-sig") as handle:
        detailed_rows = list(csv.DictReader(handle))
    cohorts, pending = aggregate_campaign(detailed_rows)

    cohort_csv = output_dir / "poster_acquisition_cohorts.csv"
    pending_csv = output_dir / "poster_pending_stage_actions.csv"
    write_csv(cohort_csv, cohorts, [
        "cycle", "peptoid_id", "strategies", "classification",
        "valid_analysed_systems", "preferred_observation_synced",
        "complete_physical_stages_of_4", "missing_physical_stages", "recommended_actions",
    ])
    write_csv(pending_csv, pending, [
        "cycle", "strategies", "peptoid_id", "system", "stage", "stage_state",
        "completed_ns", "checkpoint_exists", "current_job_ids", "action_class", "recommended_action",
        "action_reason", "directory",
    ])

    script_rows = [row for path in ML.rglob("*.py") if (row := script_metadata(path)) is not None]
    script_rows.sort(key=lambda row: natural_key(row["script_path"]))
    script_csv = output_dir / "poster_figure_script_inventory.csv"
    write_csv(script_csv, script_rows, [
        "script_path", "savefig_mentions", "plot_functions", "literal_output_names"
    ])

    figure_rows = inventory_figures(script_rows)
    figure_csv = output_dir / "poster_figure_inventory.csv"
    write_csv(figure_csv, figure_rows, [
        "poster_relevance_score", "format_class", "width_px", "height_px", "dpi_metadata",
        "size_mb", "modified_utc", "likely_source_scripts", "figure_path",
    ])

    scientific = ML / "position_resolved" / "scripts" / "scientific_figures.py"
    help_text = ""
    if scientific.is_file():
        help_run = subprocess.run(
            [sys.executable, str(scientific), "--help"], text=True, capture_output=True, check=False
        )
        help_text = (help_run.stdout or "") + (help_run.stderr or "")
    else:
        help_text = f"NOT FOUND: {scientific}\n"
    (output_dir / "scientific_figures_help.txt").write_text(help_text, encoding="utf-8")

    by_cycle: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_strategy: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in cohorts:
        cycle = int(row["cycle"])
        classification = str(row["classification"])
        by_cycle[cycle]["selected"] += 1
        by_cycle[cycle][classification] += 1
        for strategy in str(row["strategies"]).split("/"):
            if strategy:
                by_strategy[(cycle, strategy)]["selected"] += 1
                by_strategy[(cycle, strategy)][classification] += 1

    lines = [
        "PFPEPTOIDS ACTIVE-LEARNING POSTER AUDIT",
        f"Generated UTC: {utc_now()}",
        "Safety: read-only scientific inspection; only this new audit folder was written.",
        f"Output directory: {output_dir}",
        "",
        "SCIENTIFIC DEFINITIONS",
        "FULL_MEASURED = both POPE75POPG25 and POPC60CHOL40 have valid analysed 100 ns masterdata rows.",
        "SYNCED = the real paired observation is preferred in active_learning.db.",
        "The fixed 8 benchmark molecules are not relabelled as prospective active-learning tests.",
        "",
        "COHORT COUNTS BY CYCLE",
    ]
    for cycle in sorted(by_cycle):
        counts = by_cycle[cycle]
        lines.append(
            f"Cycle {cycle}: selected={counts['selected']}; "
            f"full+synced={counts['FULL_MEASURED_AND_SYNCED']}; "
            f"full masterdata not synced={counts['FULL_MEASURED_MASTERDATA_NOT_SYNCED']}; "
            f"partial one-system={counts['PARTIAL_MEASURED_ONE_SYSTEM']}; "
            f"no valid pair={counts['NO_VALID_MEASURED_PAIR']}"
        )
    lines += ["", "COHORT COUNTS BY CYCLE/STRATEGY"]
    for (cycle, strategy) in sorted(by_strategy):
        counts = by_strategy[(cycle, strategy)]
        full = counts["FULL_MEASURED_AND_SYNCED"] + counts["FULL_MEASURED_MASTERDATA_NOT_SYNCED"]
        lines.append(f"Cycle {cycle}{strategy}: selected={counts['selected']}; full measured={full}")

    lines += [
        "",
        "SIMULATION STAGE ROWS TO REVIEW BEFORE ANY SUBMISSION",
        f"Rows: {sum(row['action_class'] == 'SIMULATION_REVIEW' for row in pending)}",
    ]
    simulation_pending = [row for row in pending if row["action_class"] == "SIMULATION_REVIEW"]
    for row in simulation_pending[:100]:
        lines.append(
            f"Cycle {row['cycle']} {row['peptoid_id']} | {row['system']} {row['stage']} | "
            f"{row['stage_state']} | {row['recommended_action']} | {row['completed_ns']} ns"
        )
    if len(simulation_pending) > 100:
        lines.append(f"... {len(simulation_pending) - 100} additional rows are in {pending_csv}")
    analysis_pending = [row for row in pending if row["action_class"] == "ANALYSIS_OR_SYNC"]
    lines += [
        "",
        "ANALYSIS/SYNC ACTION ROWS (DO NOT RESUBMIT THESE AS SIMULATIONS)",
        f"Rows: {len(analysis_pending)}",
    ]
    for row in analysis_pending[:100]:
        lines.append(
            f"Cycle {row['cycle']} {row['peptoid_id']} | {row['recommended_action']} | {row['action_reason']}"
        )

    lines += [
        "",
        "TOP EXISTING FIGURE CANDIDATES (filename relevance only; visual inspection still required)",
    ]
    for row in figure_rows[:25]:
        resolution = (
            f"{row['width_px']}x{row['height_px']} px" if row["width_px"] else str(row["format_class"])
        )
        lines.append(f"score={row['poster_relevance_score']:>3} | {resolution} | {row['figure_path']}")

    lines += [
        "",
        "RECOMMENDED POSTER FIGURE",
        "Use one two-panel active-learning result figure if the completed prospective cohort is large enough:",
        "A) ranked/full-space predicted DDAPL landscape (32,768 candidates), with selected acquisitions highlighted by strategy;",
        "B) pre-simulation predicted DDAPL versus subsequently measured DDAPL for full prospective acquisitions, with y=x, n, MAE and RMSE.",
        "Do not present the fixed 8-molecule benchmark as the new prospective active-learning test cohort.",
        "If an exact existing figure does not contain both panels, regenerate only these panels from stored predictions/selection decisions/outcomes.",
        "",
        "PROVENANCE",
        f"Campaign audit script: {campaign_script}",
        f"Campaign audit exit code: {campaign_code} (0=all assertions passed; 1=lineage assertion warning)",
        f"Campaign report: {campaign_report}",
        f"Campaign detailed CSV: {campaign_csv}",
        f"Figure inventory: {figure_csv}",
        f"Figure scripts: {script_csv}",
        f"Acquisition cohorts: {cohort_csv}",
        f"Pending actions: {pending_csv}",
    ]
    report_path = output_dir / "poster_active_learning_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("PFPEPTOIDS POSTER AUDIT COMPLETE")
    print(f"Report:              {report_path}")
    print(f"Acquisition cohorts: {cohort_csv}")
    print(f"Pending actions:     {pending_csv}")
    print(f"Figure inventory:    {figure_csv}")
    print(f"Figure scripts:      {script_csv}")
    print(f"Campaign report:     {campaign_report}")
    print("No simulations, APL analyses, database rows, model files, selections, or Slurm jobs were changed.")
    # A campaign exit code of 1 means its assertions found a warning.  The poster
    # audit itself still completed and exposes the warning in both reports.
    return 0 if campaign_code in {0, 1} else campaign_code


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
