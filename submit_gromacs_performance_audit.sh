#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=pfpep-performance-audit
#SBATCH --account=tuttle-rmss
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/simulation_performance_audits/slurm_logs/pfpep-performance-audit-%j.log

set -euo pipefail

PROJECT_ROOT="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
OUTPUT_PARENT="${PROJECT_ROOT}/analysis/simulation_performance_audits"
PYTHON_EXECUTABLE="/users/pfb19164/.conda/envs/my_env_2025/bin/python"
AUDIT_SCRIPT="${SLURM_SUBMIT_DIR}/audit_gromacs_simulation_performance.py"

if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
    echo "ERROR: Python executable not found or not executable: ${PYTHON_EXECUTABLE}" >&2
    exit 2
fi

if [[ ! -f "${AUDIT_SCRIPT}" ]]; then
    echo "ERROR: Audit script not found: ${AUDIT_SCRIPT}" >&2
    echo "Both scripts must be in the directory from which sbatch is submitted." >&2
    exit 2
fi

echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Submission directory: ${SLURM_SUBMIT_DIR}"
echo "Audit script: ${AUDIT_SCRIPT}"
echo "Python executable: ${PYTHON_EXECUTABLE}"
echo "Output parent: ${OUTPUT_PARENT}"

"${PYTHON_EXECUTABLE}" "${AUDIT_SCRIPT}" \
    --root "${PROJECT_ROOT}" \
    --output-parent "${OUTPUT_PARENT}" \
    --slow-threshold 0.80 \
    --minimum-comparison-group 3
