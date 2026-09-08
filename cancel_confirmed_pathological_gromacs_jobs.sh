#!/bin/bash

# Run this script on the Archie login node with:
#   bash cancel_confirmed_pathological_gromacs_jobs.sh
# Do not submit it with sbatch.

set -uo pipefail

PROJECT_ROOT="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
AUDIT_DIR="${PROJECT_ROOT}/analysis/simulation_performance_audits"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
AUDIT_LOG="${AUDIT_DIR}/cancelled_pathological_jobs_${RUN_TIMESTAMP}.txt"

mkdir -p "${AUDIT_DIR}"

# Only these job IDs are eligible for cancellation. They were identified in
# the 2026-09-04 live diagnostic as pathological jobs on affected nodes.
declare -A EXPECTED_NAME=(
    [8016148]="Nk-Nw-Nk_80-Pull-POPE75POPG25_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8016273]="Nm-Nse-Nfn_80-Pull-POPC60CHOL40_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8016253]="Ne-Nm-Nfn_80-Pull-POPC60CHOL40_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8016390]="Nk-Nfnap-Nl_80-Relax-POPE75POPG25_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8016391]="Nk-Nv-Nwe_80-Relax-POPC60CHOL40_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8016379]="Ni-Nfnap-Nl_80-Relax-POPC60CHOL40_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8016380]="Ni-Nfnap-Nl_80-Relax-POPE75POPG25_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8016381]="Nk-Nfex-Nfnap_80-Relax-POPC60CHOL40_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8016382]="Nk-Nfex-Nfnap_80-Relax-POPE75POPG25_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8016372]="Nfex-Nfnap-Nk_80-Relax-POPC60CHOL40_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8016368]="Nfe-Nfnap-Nl_80-Relax-POPC60CHOL40_W_WF-W_WF_runML_cycle_02-FF2.1"
    [8024748]="Ne-Nm-Nfn_80-Pull-POPC60CHOL40_W_WF-W_WF_runML_cycle_01-FF2.1"
    [8025805]="Nk-Nk-Nwe_80-Relax-POPE75POPG25_W_WF-W_WF_runML_cycle_01-FF2.1"
    [8025796]="Nn-NfeB4-Nwe_80-Relax-POPC60CHOL40_W_WF-W_WF_runML_cycle_01-FF2.1"
    [8025797]="Nn-NfeB4-Nwe_80-Relax-POPE75POPG25_W_WF-W_WF_runML_cycle_01-FF2.1"
)

CANDIDATE_IDS=(
    8016148 8016273 8016253 8016390 8016391
    8016379 8016380 8016381 8016382 8016372
    8016368 8024748 8025805 8025796 8025797
)

# These known healthy jobs are explicitly protected and are not present in the
# cancellation allowlist.
PROTECTED_HEALTHY_IDS=(
    8027060 8027061 8027063 8027064 8027065 8027067 8027068
)

AFFECTED_NODE_REGEX='^node(036|039|042|043)$'
TARGET_IDS=()

is_protected_id() {
    local candidate="$1"
    local protected
    for protected in "${PROTECTED_HEALTHY_IDS[@]}"; do
        [[ "${candidate}" == "${protected}" ]] && return 0
    done
    return 1
}

extract_field() {
    local record="$1"
    local field="$2"
    sed -n "s/.* ${field}=\([^ ]*\).*/\1/p" <<< "${record}"
}

{
    echo "PFPEPTOIDS PATHOLOGICAL JOB CANCELLATION PREFLIGHT"
    echo "Generated: $(date --iso-8601=seconds)"
    echo
    printf "%-10s %-12s %-10s %-5s %-9s %-12s %s\n" \
        "JOBID" "STATE" "NODE" "CPUS" "PHASE" "CPT_AGE_MIN" "DECISION"
    printf '%s\n' "----------------------------------------------------------------------------------------------------"
} | tee "${AUDIT_LOG}"

for JOB_ID in "${CANDIDATE_IDS[@]}"; do
    if is_protected_id "${JOB_ID}"; then
        printf "%-10s %-12s %-10s %-5s %-9s %-12s %s\n" \
            "${JOB_ID}" "UNKNOWN" "UNKNOWN" "?" "?" "?" \
            "SKIP: protected healthy ID" | tee -a "${AUDIT_LOG}"
        continue
    fi

    JOB_INFO="$(scontrol show job -o "${JOB_ID}" 2>/dev/null || true)"
    if [[ -z "${JOB_INFO}" ]]; then
        printf "%-10s %-12s %-10s %-5s %-9s %-12s %s\n" \
            "${JOB_ID}" "NOT_ACTIVE" "-" "-" "-" "-" \
            "SKIP: no active Slurm job" | tee -a "${AUDIT_LOG}"
        continue
    fi

    STATE="$(extract_field "${JOB_INFO}" "JobState")"
    NODE="$(extract_field "${JOB_INFO}" "NodeList")"
    CPUS="$(extract_field "${JOB_INFO}" "NumCPUs")"
    JOB_NAME="$(extract_field "${JOB_INFO}" "JobName")"
    WORKDIR="$(extract_field "${JOB_INFO}" "WorkDir")"

    if [[ "${JOB_NAME}" == *"-Pull-"* ]]; then
        PHASE="Pull"
    elif [[ "${JOB_NAME}" == *"-Relax-"* ]]; then
        PHASE="Relax"
    else
        PHASE="UNKNOWN"
    fi

    DECISION="ELIGIBLE"

    if [[ "${STATE}" != "RUNNING" ]]; then
        DECISION="SKIP: state is not RUNNING"
    elif [[ "${JOB_NAME}" != "${EXPECTED_NAME[${JOB_ID}]}" ]]; then
        DECISION="SKIP: job-name mismatch"
    elif [[ ! "${NODE}" =~ ${AFFECTED_NODE_REGEX} ]]; then
        DECISION="SKIP: node is not an affected node"
    elif [[ "${CPUS}" != "8" ]]; then
        DECISION="SKIP: allocation is not 8 CPUs"
    elif [[ "${PHASE}" == "UNKNOWN" ]]; then
        DECISION="SKIP: phase could not be identified"
    elif [[ "${WORKDIR}" != "${PROJECT_ROOT}/Boxes/2.1/"* ]]; then
        DECISION="SKIP: work directory is outside expected project root"
    elif [[ "${WORKDIR}" != *"/run_ML_cycle_01/"* && \
            "${WORKDIR}" != *"/run_ML_cycle_02/"* ]]; then
        DECISION="SKIP: work directory is not Cycle 1 or Cycle 2"
    fi

    CHECKPOINT="${WORKDIR}/${PHASE}.cpt"
    CHECKPOINT_AGE="N/A"

    if [[ "${DECISION}" == "ELIGIBLE" ]]; then
        if [[ ! -s "${CHECKPOINT}" ]]; then
            DECISION="SKIP: checkpoint missing or empty"
        else
            CHECKPOINT_MTIME="$(stat -c %Y "${CHECKPOINT}")"
            CHECKPOINT_AGE="$(( ($(date +%s) - CHECKPOINT_MTIME) / 60 ))"
            TARGET_IDS+=("${JOB_ID}")
        fi
    fi

    printf "%-10s %-12s %-10s %-5s %-9s %-12s %s\n" \
        "${JOB_ID}" "${STATE:-UNKNOWN}" "${NODE:-UNKNOWN}" \
        "${CPUS:-?}" "${PHASE}" "${CHECKPOINT_AGE}" "${DECISION}" \
        | tee -a "${AUDIT_LOG}"
done

echo | tee -a "${AUDIT_LOG}"

if (( ${#TARGET_IDS[@]} == 0 )); then
    echo "No jobs passed every cancellation safeguard. Nothing was cancelled." \
        | tee -a "${AUDIT_LOG}"
    echo "Audit log: ${AUDIT_LOG}"
    exit 0
fi

echo "Exact jobs that passed every safeguard:" | tee -a "${AUDIT_LOG}"
printf '  %s\n' "${TARGET_IDS[@]}" | tee -a "${AUDIT_LOG}"

echo
echo "The following healthy jobs are protected and will not be cancelled:"
printf '  %s\n' "${PROTECTED_HEALTHY_IDS[@]}"

echo
read -r -p "Type CANCEL_CONFIRMED_BROKEN to cancel only the eligible jobs: " CONFIRMATION

if [[ "${CONFIRMATION}" != "CANCEL_CONFIRMED_BROKEN" ]]; then
    echo "Confirmation not received. Nothing was cancelled." | tee -a "${AUDIT_LOG}"
    echo "Audit log: ${AUDIT_LOG}"
    exit 0
fi

echo | tee -a "${AUDIT_LOG}"
echo "Cancellation requested: $(date --iso-8601=seconds)" | tee -a "${AUDIT_LOG}"

for JOB_ID in "${TARGET_IDS[@]}"; do
    if scancel "${JOB_ID}"; then
        echo "CANCEL REQUEST SUCCEEDED: ${JOB_ID}" | tee -a "${AUDIT_LOG}"
    else
        echo "CANCEL REQUEST FAILED: ${JOB_ID}" | tee -a "${AUDIT_LOG}"
    fi
done

sleep 5

echo | tee -a "${AUDIT_LOG}"
echo "Final accounting state:" | tee -a "${AUDIT_LOG}"
sacct -X -n -P -j "$(IFS=,; echo "${TARGET_IDS[*]}")" \
    -o JobIDRaw,JobName%100,State,ExitCode,Elapsed,AllocCPUS,NodeList \
    | tee -a "${AUDIT_LOG}"

echo
echo "Audit log: ${AUDIT_LOG}"
