#!/bin/bash
#SBATCH --job-name=check_ML_cycles
#SBATCH --output=check_ML_cycles_%j.out
#SBATCH --error=check_ML_cycles_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

ROOT1="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1/POPC60CHOL40_W_WF/run_ML_cycle_02"
ROOT2="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1/POPE75POPG25_W_WF/run_ML_cycle_02"

OUT="ML_cycle_02_status.csv"

echo '"system","stage","relative_path","step","sim_time_ps","sim_time_ns","hours_running","last_modified","pull_log"' > "$OUT"

for ROOT in "$ROOT1" "$ROOT2"; do

    SYSTEM=$(basename "$(dirname "$ROOT")")

    find "$ROOT" -type f -name "Pull.log" -print0 |
    while IFS= read -r -d '' LOG; do

        DIR=$(dirname "$LOG")

        # Directory containing Pull.log
        STAGE=$(basename "$DIR")

        # Path relative to run_ML_cycle_02
        RELPATH="${DIR#$ROOT/}"

        # ----------------------------------------------------------
        # Extract latest Step and Time from the GROMACS log
        #
        # Example:
        #       Step           Time
        #    4290000    85800.00000
        # ----------------------------------------------------------
        read STEP SIM_TIME_PS <<< "$(awk '
            /^[[:space:]]*Step[[:space:]]+Time/ {
                getline
                step=$1
                time=$2
            }
            END {
                print step, time
            }
        ' "$LOG")"

        if [[ -z "$STEP" ]]; then
            STEP="NA"
        fi

        if [[ -n "$SIM_TIME_PS" ]]; then
            SIM_TIME_NS=$(awk -v t="$SIM_TIME_PS" 'BEGIN {printf "%.3f", t/1000}')
        else
            SIM_TIME_PS="NA"
            SIM_TIME_NS="NA"
        fi

        # ----------------------------------------------------------
        # Determine wall-clock running time
        #
        # First try GROMACS "Started mdrun..." line.
        # ----------------------------------------------------------
        START_LINE=$(grep -m1 "Started mdrun" "$LOG")

        HOURS="NA"

        if [[ -n "$START_LINE" ]]; then

            # Strip everything before the timestamp
            START_DATE=$(echo "$START_LINE" |
                         sed -E 's/^.*Started mdrun[^A-Za-z]*(on rank [0-9]+ )?//')

            START_EPOCH=$(date -d "$START_DATE" +%s 2>/dev/null)

            if [[ -n "$START_EPOCH" ]]; then
                NOW=$(date +%s)

                HOURS=$(awk \
                    -v now="$NOW" \
                    -v start="$START_EPOCH" \
                    'BEGIN {printf "%.2f", (now-start)/3600}')
            fi
        fi

        # ----------------------------------------------------------
        # Fallback: filesystem creation time
        # ----------------------------------------------------------
        if [[ "$HOURS" == "NA" ]]; then

            BIRTH=$(stat -c %W "$LOG" 2>/dev/null)

            if [[ -n "$BIRTH" && "$BIRTH" -gt 0 ]]; then
                NOW=$(date +%s)

                HOURS=$(awk \
                    -v now="$NOW" \
                    -v start="$BIRTH" \
                    'BEGIN {printf "%.2f", (now-start)/3600}')
            fi
        fi

        MODIFIED=$(stat -c '%y' "$LOG" | cut -d'.' -f1)

        # Escape quotes for CSV
        SYSTEM=${SYSTEM//\"/\"\"}
        STAGE=${STAGE//\"/\"\"}
        RELPATH=${RELPATH//\"/\"\"}
        LOG_ESC=${LOG//\"/\"\"}

        printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
            "$SYSTEM" \
            "$STAGE" \
            "$RELPATH" \
            "$STEP" \
            "$SIM_TIME_PS" \
            "$SIM_TIME_NS" \
            "$HOURS" \
            "$MODIFIED" \
            "$LOG_ESC" >> "$OUT"

    done
done

echo
echo "========================================================"
echo "Scan complete"
echo "========================================================"
echo
echo "CSV written to:"
echo "$OUT"
echo

column -s, -t "$OUT"
