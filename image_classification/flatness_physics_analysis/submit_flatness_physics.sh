#!/bin/bash
# Usage:
# bash submit_flatness_physics.sh [masterdataanalysis.csv] [system_morphology.csv] [canonical_annotations.csv] [output_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
ANALYSIS="$PROJECT/analysis"
MORPH_ROOT="$ANALYSIS/image_classification/morph_prod"

MASTER="${1:-$ANALYSIS/masterdataanalysis.csv}"

if [ -n "${2:-}" ]; then
  MORPH="$2"
else
  RUN_TAG="$(cat "$MORPH_ROOT/qc/current_production_tag.txt" 2>/dev/null || true)"
  CAND="$MORPH_ROOT/results/production_collection_${RUN_TAG}/system_morphology.csv"
  if [ -f "$CAND" ]; then
    MORPH="$CAND"
  else
    MORPH="$(find "$MORPH_ROOT/results" "$MORPH_ROOT" -path '*/production_collection_*/system_morphology.csv' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  fi
fi

ANNOT="${3:-$ANALYSIS/image_classification/pfp34_new_photos/session/canonical_export_20260929/canonical_annotations.csv}"
TAG="$(date +%Y%m%d_%H%M%S)"
OUT="${4:-$SCRIPT_DIR/results_flatness_physics_${TAG}}"
PYTHON="/users/pfb19164/.conda/envs/my_env_2025/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python)"

for f in "$MASTER" "$MORPH" "$ANNOT" "$SCRIPT_DIR/flatness_physics_analysis.py" "$SCRIPT_DIR/flatness_physics_summarize.py"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: required file not found: $f" >&2
    exit 2
  fi
done

mkdir -p "$OUT/logs"

echo "MASTER=$MASTER"
echo "MORPH=$MORPH"
echo "ANNOT=$ANNOT"
echo "OUT=$OUT"
echo "PYTHON=$PYTHON"

ARRAY_JOB="$(sbatch --parsable \
  --output="$OUT/logs/array_%A_%a.log" \
  --error="$OUT/logs/array_%A_%a.err" \
  --export=ALL,MASTER="$MASTER",MORPH="$MORPH",ANNOT="$ANNOT",OUT="$OUT",SCRIPT_DIR="$SCRIPT_DIR",PYTHON="$PYTHON" \
  "$SCRIPT_DIR/flatness_physics_array.sbatch")"

SUMMARY_JOB="$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_JOB" \
  --output="$OUT/logs/summary_%j.log" \
  --error="$OUT/logs/summary_%j.err" \
  --export=ALL,OUT="$OUT",SCRIPT_DIR="$SCRIPT_DIR",PYTHON="$PYTHON" \
  "$SCRIPT_DIR/flatness_physics_summary.sbatch")"

cat > "$OUT/submission.txt" <<EOF
ARRAY_JOB=$ARRAY_JOB
SUMMARY_JOB=$SUMMARY_JOB
MASTER=$MASTER
MORPH=$MORPH
ANNOT=$ANNOT
OUT=$OUT
EOF

echo
echo "Submitted analysis array: $ARRAY_JOB (7 parallel 1-CPU tasks)"
echo "Submitted dependent summary: $SUMMARY_JOB"
echo
echo "Monitor:"
echo "  squeue -j $ARRAY_JOB,$SUMMARY_JOB"
echo
echo "When complete:"
echo "  cat '$OUT/MEETING_SUMMARY.md'"
echo
echo "All outputs:"
echo "  find '$OUT' -maxdepth 2 -type f | sort"
