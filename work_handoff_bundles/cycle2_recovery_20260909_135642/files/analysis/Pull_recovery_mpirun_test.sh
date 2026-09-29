#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=Nk-Nw-Nwe_80-Pull-recovery-mpirun
#SBATCH --account=tuttle-rmss
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --exclude=node036,node039,node042,node043
#SBATCH --output=/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1/POPC60CHOL40_W_WF/run_ML_cycle_02/Nk-Nw-Nwe_80/Slurm-recovery-mpirun-%j.log

# This is a controlled two-hour recovery/performance test. It resumes the
# existing Pull simulation and stops cleanly after about 1.75 hours.
set -eo pipefail

module purge
module load gromacs/gcc-14.2.1/2025.3

set -u

WORKDIR="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1/POPC60CHOL40_W_WF/run_ML_cycle_02/Nk-Nw-Nwe_80"
TABLE="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/angle5_a0.xvg"
JOB_PREFIX="Nk-Nw-Nwe_80-Pull-POPC60CHOL40"

cd "${WORKDIR}"

for required_file in Pull.tpr Pull.cpt "${TABLE}"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "ERROR: required file is absent or empty: ${required_file}" >&2
        exit 1
    fi
done

# Refuse to start if another queued/running job appears to target this same
# peptoid, phase and bilayer. Ignore this job's own Slurm record.
while IFS='|' read -r other_job_id other_job_name other_state; do
    [[ -z "${other_job_id}" ]] && continue
    [[ "${other_job_id}" == "${SLURM_JOB_ID}" ]] && continue
    if [[ "${other_job_name}" == "${JOB_PREFIX}"* ]]; then
        echo "ERROR: possible overlapping job detected:" >&2
        echo "  ${other_job_id} | ${other_state} | ${other_job_name}" >&2
        exit 1
    fi
done < <(squeue -u "${USER}" -h -o "%A|%.250j|%T")

BACKUP_DIR="${WORKDIR}/pre_recovery_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"
cp -p Pull.cpt "${BACKUP_DIR}/"
if [[ -s Pull_prev.cpt ]]; then
    cp -p Pull_prev.cpt "${BACKUP_DIR}/"
fi

echo "Recovery job: ${SLURM_JOB_ID}"
echo "Node: $(hostname -f)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Working directory: ${WORKDIR}"
echo "Checkpoint: ${WORKDIR}/Pull.cpt"
echo "Backup directory: ${BACKUP_DIR}"
echo "Started: $(date --iso-8601=seconds)"

/opt/software/scripts/job_prologue.sh

export OMP_NUM_THREADS=8
export OMP_DYNAMIC=false
export OMP_PLACES=cores
export OMP_PROC_BIND=close

mpirun -np 1 \
    --map-by "ppr:1:node:PE=8" \
    --bind-to core \
    --report-bindings \
    gmx_mpi mdrun \
    -ntomp 8 \
    -pin off \
    -tableb "${TABLE}" \
    -deffnm Pull \
    -cpi Pull.cpt \
    -append \
    -px Pull_pullx.xvg \
    -pf Pull_pullf.xvg \
    -cpt 15 \
    -maxh 1.75

/opt/software/scripts/job_epilogue.sh

echo "Finished: $(date --iso-8601=seconds)"
