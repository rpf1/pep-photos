#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=Ni-Nfnap-Nl_80-Relax-POPC60CHOL40_W_WF-C2-recovery
#SBATCH --account=tuttle-rmss
#SBATCH --partition=standard
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --exclude=node036,node039,node042,node043
#SBATCH --output=/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1/POPC60CHOL40_W_WF/run_ML_cycle_02/Ni-Nfnap-Nl_80/Slurm-C2-recovery-Relax-%j.log

set -euo pipefail

module purge
module load gromacs/gcc-14.2.1/2025.3

WORKDIR="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1/POPC60CHOL40_W_WF/run_ML_cycle_02/Ni-Nfnap-Nl_80"
TABLE="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/angle5_a0.xvg"
STAGE="Relax"

cd "${WORKDIR}"

for required_file in "${STAGE}.tpr" "${STAGE}.cpt" "${TABLE}"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "ERROR: required file absent/empty: ${required_file}" >&2
        exit 1
    fi
done

# Refuse to overlap another job for the same peptoid/stage/bilayer.
while IFS='|' read -r other_job_id other_job_name other_state; do
    [[ -z "${other_job_id}" ]] && continue
    [[ "${other_job_id}" == "${SLURM_JOB_ID}" ]] && continue

    lower_name="$(printf '%s' "${other_job_name}" | tr '[:upper:]' '[:lower:]')"

    if [[ "${lower_name}" == *"ni-nfnap-nl"* ]]        && [[ "${lower_name}" == *"relax"* ]]        && [[ "${lower_name}" == *"popc60chol40_w_wf"* ]]; then
        echo "ERROR: possible overlapping job detected:" >&2
        echo "  ${other_job_id} | ${other_state} | ${other_job_name}" >&2
        exit 1
    fi
done < <(squeue -u "${USER}" -h -o "%A|%.250j|%T")

# Back up the checkpoint immediately before continuation.
BACKUP_DIR="${WORKDIR}/pre_C2_recovery_${STAGE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"

cp -p "${STAGE}.cpt" "${BACKUP_DIR}/"

if [[ -s "${STAGE}_prev.cpt" ]]; then
    cp -p "${STAGE}_prev.cpt" "${BACKUP_DIR}/"
fi

echo "Recovery job: ${SLURM_JOB_ID}"
echo "Node: $(hostname -f)"
echo "Peptoid: Ni-Nfnap-Nl"
echo "Bilayer: POPC60CHOL40_W_WF"
echo "Stage: ${STAGE}"
echo "Working directory: ${WORKDIR}"
echo "Checkpoint: ${WORKDIR}/${STAGE}.cpt"
echo "Backup directory: ${BACKUP_DIR}"
echo "Started: $(date --iso-8601=seconds)"

/opt/software/scripts/job_prologue.sh

export OMP_NUM_THREADS=8
export OMP_DYNAMIC=false
export OMP_PLACES=cores
export OMP_PROC_BIND=close

mpirun -np 1     --map-by "ppr:1:node:PE=8"     --bind-to core     --report-bindings     gmx_mpi mdrun     -ntomp 8     -pin off     -tableb "${TABLE}"     -deffnm "${STAGE}"     -cpi "${STAGE}.cpt"     -append     -cpt 15     -maxh 23.5

/opt/software/scripts/job_epilogue.sh

echo "Finished: $(date --iso-8601=seconds)"
