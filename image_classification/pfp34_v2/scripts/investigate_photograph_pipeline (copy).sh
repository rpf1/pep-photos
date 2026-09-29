#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# PFPeptoids photograph pipeline — read-only investigation
#
# Purpose:
#   1. Locate current Photograph-related source files.
#   2. Inspect current image/view generation.
#   3. Inspect VMD rotations used for side views.
#   4. Inspect peptoid bead/residue selections.
#   5. Inspect STACKED/composite-image construction.
#   6. Optionally inspect one problematic .gro file and flag
#      residue names that do not appear anywhere in the current
#      photograph source code.
#
# Usage:
#   bash investigate_photograph_pipeline.sh
#
# Optional problematic GRO:
#   bash investigate_photograph_pipeline.sh /full/path/system.gro
#
# Output:
#   Prints report to terminal only.
#
# READ-ONLY:
#   Does not edit, delete, move, rename, or regenerate anything.
# ============================================================

ROOT="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
ANALYSIS="${ROOT}/analysis"

BROKEN_GRO="${1:-}"

echo "============================================================"
echo "PFPEPTOIDS PHOTOGRAPH PIPELINE — READ-ONLY INVESTIGATION"
echo "============================================================"
echo
echo "Project root:"
echo "  ${ROOT}"
echo
echo "Analysis root:"
echo "  ${ANALYSIS}"
echo

if [[ ! -d "${ANALYSIS}" ]]; then
    echo "ERROR: analysis directory does not exist:"
    echo "  ${ANALYSIS}"
    exit 1
fi


# ------------------------------------------------------------
# 1. Find current photograph-related files
# ------------------------------------------------------------

echo
echo "============================================================"
echo "1. PHOTOGRAPH-RELATED FILES"
echo "============================================================"

mapfile -t PHOTO_FILES < <(
    find "${ROOT}" \
        -type f \
        \( \
            -iname 'photograph.py' \
            -o -iname 'photgraph.py' \
            -o -iname 'photograph.tcl' \
            -o -iname 'photograph_components.py' \
            -o -iname 'photograph_components.tcl' \
            -o -iname '*photograph*.py' \
            -o -iname '*photograph*.tcl' \
        \) \
        2>/dev/null | sort
)

if (( ${#PHOTO_FILES[@]} == 0 )); then
    echo "No Photograph-related Python/TCL files found."
    exit 1
fi

printf '%s\n' "${PHOTO_FILES[@]}"

echo
echo "File metadata:"
for f in "${PHOTO_FILES[@]}"; do
    echo
    echo "--- ${f}"
    stat -c 'size=%s bytes | modified=%y' "${f}" 2>/dev/null || ls -lh "${f}"
done


# ------------------------------------------------------------
# 2. Find current raw-image names / image outputs
# ------------------------------------------------------------

echo
echo "============================================================"
echo "2. IMAGE FILENAMES / OUTPUT VIEWS REFERENCED IN SOURCE"
echo "============================================================"

grep -nEi \
    'snapshot|stacked|top|bottom|side|water|bilayer|\.png|\.jpg|\.jpeg' \
    "${PHOTO_FILES[@]}" \
    2>/dev/null || true


# ------------------------------------------------------------
# 3. Examine VMD orientation / rotation commands
# ------------------------------------------------------------

echo
echo "============================================================"
echo "3. CAMERA / ROTATION / ORIENTATION LOGIC"
echo "============================================================"

grep -nEi \
    'rotate|rot[[:space:]_]*(x|y|z)|transaxis|viewpoint|camera|display[[:space:]]+resetview|resetview|render' \
    "${PHOTO_FILES[@]}" \
    2>/dev/null || true


# ------------------------------------------------------------
# 4. Examine molecule selections / bead / residue logic
# ------------------------------------------------------------

echo
echo "============================================================"
echo "4. RESIDUE / BEAD / VMD SELECTION LOGIC"
echo "============================================================"

grep -nEi \
    'resname|resid|name[[:space:]]|atomselect|selection|select|specialindex|index|peptoid|bead|vdw|representation|mol modselect|mol selection' \
    "${PHOTO_FILES[@]}" \
    2>/dev/null || true


# ------------------------------------------------------------
# 5. Explicitly search known/suspicious aliases
# ------------------------------------------------------------

echo
echo "============================================================"
echo "5. KNOWN / SUSPICIOUS RESIDUE-NAME ALIASES"
echo "============================================================"

ALIASES=(
    Na
    Nx
    Nab
    Ni
    Nl
    Nm
    Nv
    Nn
    Nq
    Ns
    Nse
    Nt
    NmO
    Nd
    Ne
    Nf
    Nfn
    Nfe
    Nfex
    Nw
    Nwe
    Ny
    Nfe4Cl
    NfeC4
    Nfe4Br
    NfeB4
    Nfnaph
    Nfnap
    Nk
    Nke
    Nkeqm
    Nr
)

for alias in "${ALIASES[@]}"; do
    hits="$(
        grep -Fwn "${alias}" "${PHOTO_FILES[@]}" 2>/dev/null || true
    )"

    if [[ -n "${hits}" ]]; then
        echo
        echo "[FOUND] ${alias}"
        echo "${hits}"
    else
        echo "[NOT FOUND] ${alias}"
    fi
done


# ------------------------------------------------------------
# 6. Show context around potentially important functions
# ------------------------------------------------------------

echo
echo "============================================================"
echo "6. IMPORTANT FUNCTION / CODE LOCATIONS"
echo "============================================================"

grep -nEi \
    'def .*photo|def .*image|def .*stack|def .*combine|def .*trim|def .*crop|def .*special|def .*index|def .*render|def .*snapshot|proc .*photo|proc .*render' \
    "${PHOTO_FILES[@]}" \
    2>/dev/null || true


# ------------------------------------------------------------
# 7. Image-combination / PIL logic
# ------------------------------------------------------------

echo
echo "============================================================"
echo "7. STACKING / CROPPING / COMPOSITE IMAGE LOGIC"
echo "============================================================"

grep -nEi \
    'PIL|Image\.open|Image\.new|crop|trim|resize|paste|concatenate|stack|combine|append|hstack|vstack|canvas|width|height' \
    "${PHOTO_FILES[@]}" \
    2>/dev/null || true


# ------------------------------------------------------------
# 8. Search for existing Y-side support
# ------------------------------------------------------------

echo
echo "============================================================"
echo "8. EXISTING SECOND-SIDE / Y-SIDE SUPPORT"
echo "============================================================"

grep -nEi \
    'side.?y|y.?side|snapshot_.*y|rotate[[:space:]]+y|transaxis[[:space:]]+y|rot[[:space:]_]*y' \
    "${PHOTO_FILES[@]}" \
    2>/dev/null || true


# ------------------------------------------------------------
# 9. Optional GRO inspection
# ------------------------------------------------------------

if [[ -n "${BROKEN_GRO}" ]]; then

    echo
    echo "============================================================"
    echo "9. PROBLEMATIC GRO INSPECTION"
    echo "============================================================"

    if [[ ! -f "${BROKEN_GRO}" ]]; then
        echo "ERROR: supplied GRO file does not exist:"
        echo "  ${BROKEN_GRO}"
        exit 1
    fi

    echo "GRO:"
    echo "  ${BROKEN_GRO}"
    echo

    echo "Unique residue names in GRO:"
    GRO_RESNAMES="$(
        sed '1,2d;$d' "${BROKEN_GRO}" \
        | awk '
            length($0) >= 10 {
                r = substr($0, 6, 5)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", r)
                if (r != "") print r
            }
        ' \
        | sort -u
    )"

    echo "${GRO_RESNAMES}"

    echo
    echo "Residue counts:"
    sed '1,2d;$d' "${BROKEN_GRO}" \
    | awk '
        length($0) >= 10 {
            resid = substr($0, 1, 5)
            resname = substr($0, 6, 5)

            gsub(/^[[:space:]]+|[[:space:]]+$/, "", resid)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", resname)

            key = resid ":" resname

            if (!(key in seen)) {
                seen[key] = 1
                count[resname]++
            }
        }

        END {
            for (r in count) {
                printf "%-12s %d residues\n", r, count[r]
            }
        }
    ' | sort

    echo
    echo "------------------------------------------------------------"
    echo "N*-prefixed residue names versus photograph source"
    echo "------------------------------------------------------------"

    while IFS= read -r residue; do
        [[ -z "${residue}" ]] && continue

        # Peptoid residue names in this project normally begin N.
        if [[ "${residue}" == N* ]]; then

            if grep -Fqw "${residue}" "${PHOTO_FILES[@]}" 2>/dev/null; then
                echo "[SOURCE CONTAINS] ${residue}"
            else
                echo "[POTENTIAL MISMATCH] ${residue}"
            fi
        fi
    done <<< "${GRO_RESNAMES}"

else

    echo
    echo "============================================================"
    echo "9. PROBLEMATIC GRO INSPECTION"
    echo "============================================================"
    echo "Skipped: no GRO file supplied."
    echo
    echo "Once you identify a broken system, rerun as:"
    echo
    echo "  bash $0 /full/path/to/problematic/system.gro"
fi


# ------------------------------------------------------------
# 10. Search Photograph source for literal residue-like tokens
# ------------------------------------------------------------

echo
echo "============================================================"
echo "10. RESIDUE-LIKE TOKENS PRESENT IN PHOTOGRAPH SOURCE"
echo "============================================================"

grep -Eho '\bN[A-Za-z][A-Za-z0-9]*\b' "${PHOTO_FILES[@]}" 2>/dev/null \
    | sort -u \
    || true


# ------------------------------------------------------------
# 11. Summary
# ------------------------------------------------------------

echo
echo "============================================================"
echo "INVESTIGATION COMPLETE"
echo "============================================================"
echo
echo "Please send me the COMPLETE terminal output from this script."
echo
echo "Most important sections:"
echo "  1. Photograph-related files"
echo "  3. rotation/orientation logic"
echo "  4. residue/bead selections"
echo "  5. suspicious aliases"
echo "  7. stacking/composite logic"
echo "  8. Y-side support"
echo "  9. problematic GRO inspection, if available"
echo
echo "No files were modified."
