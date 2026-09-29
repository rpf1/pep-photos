#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# PFPeptoids photograph pipeline — TARGETED READ-ONLY investigation
#
# Target:
#   Nfnap-NfeC4-NfeC4
#   POPE75POPG25_W_WF
#   run_0
#
# No files are modified.
# ============================================================

ROOT="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
ANALYSIS="${ROOT}/analysis"

MOLECULE="Nfnap-NfeC4-NfeC4"
BILAYER="POPE75POPG25_W_WF"
RUN="run_0"

SYSTEM_DIR="${ROOT}/Boxes/2.1/${BILAYER}/${RUN}/${MOLECULE}_80"

SOURCE_FILES=(
    "${ANALYSIS}/Photograph.py"
    "${ANALYSIS}/Photograph.tcl"
    "${ANALYSIS}/Photograph_components.py"
    "${ANALYSIS}/Photograph_components.tcl"
)

echo "============================================================"
echo "PFPEPTOIDS PHOTOGRAPH PIPELINE — TARGETED INVESTIGATION"
echo "============================================================"
echo
echo "Molecule : ${MOLECULE}"
echo "Bilayer  : ${BILAYER}"
echo "Run      : ${RUN}"
echo "System   : ${SYSTEM_DIR}"
echo


# ============================================================
# 1. Verify canonical source files
# ============================================================

echo "============================================================"
echo "1. CANONICAL PHOTOGRAPH SOURCE FILES"
echo "============================================================"

for f in "${SOURCE_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "[FOUND] $f"
    else
        echo "[MISSING] $f"
    fi
done


# ============================================================
# 2. Relevant files in problematic system
# ============================================================

echo
echo "============================================================"
echo "2. TARGET SYSTEM FILES"
echo "============================================================"

if [[ ! -d "${SYSTEM_DIR}" ]]; then
    echo "ERROR: system directory not found:"
    echo "${SYSTEM_DIR}"
    exit 1
fi

find "${SYSTEM_DIR}" \
    -maxdepth 2 \
    -type f \
    \( \
        -iname '*.gro' \
        -o -iname '*.pdb' \
        -o -iname '*.xtc' \
        -o -iname '*.trr' \
        -o -iname '*.tpr' \
        -o -iname '*.tcl' \
        -o -iname '*.png' \
    \) \
    -printf '%P\n' \
    | sort


# ============================================================
# 3. Current render outputs
# ============================================================

echo
echo "============================================================"
echo "3. CURRENT SNAPSHOT / IMAGE OUTPUTS"
echo "============================================================"

grep -nEi \
    'snapshot|render|\.png|stacked' \
    "${SOURCE_FILES[@]}" \
    2>/dev/null || true


# ============================================================
# 4. Current camera / rotation logic
# ============================================================

echo
echo "============================================================"
echo "4. CURRENT CAMERA / ROTATION LOGIC"
echo "============================================================"

grep -nEi \
    'rotate|rot[[:space:]_]*(x|y|z)|transaxis|resetview|viewpoint|camera|scale' \
    "${SOURCE_FILES[@]}" \
    2>/dev/null || true


# ============================================================
# 5. Selection / representation logic
# ============================================================

echo
echo "============================================================"
echo "5. CURRENT SELECTION / REPRESENTATION LOGIC"
echo "============================================================"

grep -nEi \
    'atomselect|resname|resid|name[[:space:]]|SpecialIndex|index|mol modselect|selection|VDW|representation|peptoid' \
    "${SOURCE_FILES[@]}" \
    2>/dev/null || true


# ============================================================
# 6. Specifically inspect Nfnap / NfeC4 aliases
# ============================================================

echo
echo "============================================================"
echo "6. TARGET RESIDUE / ALIAS REFERENCES"
echo "============================================================"

for token in \
    Nfnap Nfnaph \
    NfeC4 Nfe4Cl \
    NfeB4 Nfe4Br \
    Na Nx
do
    echo
    echo "----- ${token} -----"

    grep -Fnw \
        "${token}" \
        "${SOURCE_FILES[@]}" \
        2>/dev/null || echo "NOT FOUND"
done


# ============================================================
# 7. Python functions controlling indexing / image assembly
# ============================================================

echo
echo "============================================================"
echo "7. IMPORTANT PYTHON FUNCTIONS"
echo "============================================================"

grep -nEi \
    '^def .*(' \
    "${ANALYSIS}/Photograph.py" \
    "${ANALYSIS}/Photograph_components.py" \
    2>/dev/null || true

echo
echo "Image assembly/cropping references:"
grep -nEi \
    'Image\.open|Image\.new|paste|crop|resize|stack|combine|width|height|snapshot_' \
    "${ANALYSIS}/Photograph.py" \
    "${ANALYSIS}/Photograph_components.py" \
    2>/dev/null || true


# ============================================================
# 8. Look specifically for existing Y-side support
# ============================================================

echo
echo "============================================================"
echo "8. EXISTING Y-SIDE / SECOND-SIDE SUPPORT"
echo "============================================================"

grep -nEi \
    'side.?y|y.?side|snapshot_.*y|rotate[[:space:]]+y|transaxis[[:space:]]+y|rot[[:space:]_]*y' \
    "${SOURCE_FILES[@]}" \
    2>/dev/null || echo "NO Y-SIDE IMPLEMENTATION FOUND"


# ============================================================
# 9. Inspect GRO residue + atom names
# ============================================================

echo
echo "============================================================"
echo "9. TARGET GRO CONTENTS"
echo "============================================================"

mapfile -t GRO_FILES < <(
    find "${SYSTEM_DIR}" -maxdepth 2 -type f -name '*.gro' | sort
)

if (( ${#GRO_FILES[@]} == 0 )); then
    echo "No GRO files found in ${SYSTEM_DIR}"
else

    for gro in "${GRO_FILES[@]}"; do
        echo
        echo "------------------------------------------------------------"
        echo "GRO: ${gro}"
        echo "------------------------------------------------------------"

        echo
        echo "Unique residue names:"
        sed '1,2d;$d' "${gro}" \
        | awk '
            length($0) >= 15 {
                x = substr($0,6,5)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", x)
                if (x != "") print x
            }
        ' \
        | sort -u

        echo
        echo "Unique atom/bead names:"
        sed '1,2d;$d' "${gro}" \
        | awk '
            length($0) >= 15 {
                x = substr($0,11,5)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", x)
                if (x != "") print x
            }
        ' \
        | sort -u

        echo
        echo "Target-like residues and their atom/bead names:"
        sed '1,2d;$d' "${gro}" \
        | awk '
            length($0) >= 15 {
                r = substr($0,6,5)
                a = substr($0,11,5)

                gsub(/^[[:space:]]+|[[:space:]]+$/, "", r)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", a)

                if (
                    r ~ /^Nfnap$/ ||
                    r ~ /^Nfnaph$/ ||
                    r ~ /^NfeC4$/ ||
                    r ~ /^Nfe4Cl$/
                ) {
                    print r, a
                }
            }
        ' \
        | sort -u
    done
fi


# ============================================================
# 10. Generated TCL for this exact system
# ============================================================

echo
echo "============================================================"
echo "10. GENERATED TCL FOR TARGET SYSTEM"
echo "============================================================"

mapfile -t GENERATED_TCL < <(
    find "${SYSTEM_DIR}" \
        -maxdepth 2 \
        -type f \
        \( -name 'Photograph.tcl' -o -name 'Photograph_generated.tcl' \) \
        | sort
)

if (( ${#GENERATED_TCL[@]} == 0 )); then
    echo "No generated Photograph TCL found."
else
    for tcl in "${GENERATED_TCL[@]}"; do
        echo
        echo "----- ${tcl} -----"

        grep -nEi \
            'atomselect|resname|name[[:space:]]|selection|rotate|transaxis|render snapshot|mol modselect' \
            "${tcl}" \
            2>/dev/null || true
    done
fi


# ============================================================
# DONE
# ============================================================

echo
echo "============================================================"
echo "INVESTIGATION COMPLETE"
echo "============================================================"
echo
echo "No files were modified."
