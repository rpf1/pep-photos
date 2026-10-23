#!/bin/bash
set -euo pipefail
: "${MORPH_PYTHON:?Export the absolute Python executable}"
plan="$(cd -- "$(dirname -- "$0")" && pwd)"
package="$(cd -- "$plan/../.." && pwd)"
exec "$MORPH_PYTHON" "$package/scripts/morphology_production.py" submit --plan "$plan" "$@"
