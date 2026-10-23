#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
project=/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids
output="$project/analysis/photograph_validation/pilot_$(date +%Y%m%d_%H%M%S)_$$"
python validate_photographs.py \
  --system "$project/Boxes/2.1/POPE75POPG25_W_WF/run_0/Nfnap-NfeC4-NfeC4_80" \
  --system "$project/Boxes/2.1/POPC60CHOL40_W_WF/run_0/Na-Nab-Ni_80" \
  --system "$project/Boxes/2.1/POPE75POPG25_W_WF/run_0/Nwe-NfeB4-Nke_80" \
  --system "$project/Boxes/2.1/POPE75POPG25_W_WF/run_ML_cycle_02/Nk-Nm-Nke_80" \
  --output "$output" "$@"
