#!/usr/bin/env bash
# Local impact feedback loop: validate Rivanna results, gates, H5 ablation, plan next wave.
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=0
SKIP_H5=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --skip-h5) SKIP_H5=1 ;;
  esac
done

ARGS=()
[[ "${QUICK}" -eq 1 ]] && ARGS+=(--quick)
[[ "${SKIP_H5}" -eq 1 ]] && ARGS+=(--skip-h5)

if ((${#ARGS[@]})); then
  python -m hbac.scripts.impact_feedback_loop run "${ARGS[@]}"
else
  python -m hbac.scripts.impact_feedback_loop run
fi
