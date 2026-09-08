#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=pfpep-mdp-audit
#SBATCH --account=tuttle-rmss
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/simulation_parameter_audits/slurm_logs/pfpep-mdp-audit-%j.log

set -euo pipefail

PYTHON="/users/pfb19164/.conda/envs/my_env_2025/bin/python"
SEARCH_ROOT="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1"
OUTPUT_PARENT="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/simulation_parameter_audits"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_PARENT}/audit_${RUN_TIMESTAMP}"
OUTPUT_CSV="${OUTPUT_DIR}/mdp_parameter_flags.csv"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" - "${SEARCH_ROOT}" "${OUTPUT_CSV}" <<'PY'
import csv
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

search_root = Path(sys.argv[1])
output_csv = Path(sys.argv[2])

expected_dt = Decimal("0.02")
expected_nsteps = 5_000_000

cycle_pattern = re.compile(
    r"/(?:run_ml_cycle_0?([12])|run_0?([12]))/",
    re.IGNORECASE,
)


def read_parameters(path: Path):
    values = {"dt": [], "nsteps": []}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            active_line = raw_line.split(";", 1)[0].strip()
            if not active_line or "=" not in active_line:
                continue

            key, value = active_line.split("=", 1)
            key = key.strip().lower().replace("-", "_")
            value = value.strip().split()[0] if value.strip() else ""

            if key == "dt":
                values["dt"].append((line_number, value))
            elif key in {"nsteps", "n_steps"}:
                values["nsteps"].append((line_number, value))

    return values


def validate(path: Path, cycle: str, stage: str):
    issues = []
    parameters = read_parameters(path)

    dt_entries = parameters["dt"]
    nsteps_entries = parameters["nsteps"]

    if not dt_entries:
        issues.append("dt missing")
        dt_text = ""
    else:
        dt_text = dt_entries[-1][1]
        if len({value for _, value in dt_entries}) > 1:
            issues.append("multiple conflicting dt definitions")
        try:
            if Decimal(dt_text) != expected_dt:
                issues.append(f"dt={dt_text}; expected 0.02")
        except InvalidOperation:
            issues.append(f"dt is not numeric: {dt_text!r}")

    if not nsteps_entries:
        issues.append("nsteps missing")
        nsteps_text = ""
    else:
        nsteps_text = nsteps_entries[-1][1]
        if len({value for _, value in nsteps_entries}) > 1:
            issues.append("multiple conflicting nsteps definitions")
        try:
            parsed_nsteps = Decimal(nsteps_text)
            if parsed_nsteps != expected_nsteps or parsed_nsteps != parsed_nsteps.to_integral_value():
                issues.append(f"nsteps={nsteps_text}; expected 5000000")
        except InvalidOperation:
            issues.append(f"nsteps is not numeric: {nsteps_text!r}")

    if not issues:
        return None

    return {
        "cycle": cycle,
        "stage": stage,
        "file_path": str(path),
        "dt": dt_text,
        "nsteps": nsteps_text,
        "manual_inspection_reason": "; ".join(issues),
    }


flagged_rows = []
files_checked = 0

for path in sorted(search_root.rglob("*.mdp")):
    if path.name.lower() not in {"pull.mdp", "relax.mdp"}:
        continue

    normalized_path = "/" + path.as_posix().lstrip("/")
    cycle_match = cycle_pattern.search(normalized_path)
    if not cycle_match:
        continue

    cycle = cycle_match.group(1) or cycle_match.group(2)
    stage = "Pull" if path.name.lower() == "pull.mdp" else "Relax"
    files_checked += 1

    result = validate(path, cycle, stage)
    if result is not None:
        flagged_rows.append(result)

fieldnames = [
    "cycle",
    "stage",
    "file_path",
    "dt",
    "nsteps",
    "manual_inspection_reason",
]

with output_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flagged_rows)

print(f"Files checked: {files_checked}")
print(f"Files flagged: {len(flagged_rows)}")
print(f"Flagged-files CSV: {output_csv}")
PY
