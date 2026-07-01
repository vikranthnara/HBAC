#!/usr/bin/env bash
# One-shot Rivanna bootstrap + full-scale Variant B submit.
# Run ON the login node after code is at /standard/liverobotics/hbac
set -euo pipefail

export HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"

echo "=== HBAC on-cluster setup + full-scale Variant B ==="

if [[ ! -d "${HBAC_ROOT}" ]]; then
  echo "ERROR: ${HBAC_ROOT} not found."
  echo "From your laptop, sync code first:"
  echo "  bash scripts/rivanna/sync_to_rivanna.sh"
  exit 1
fi

cd "${HBAC_ROOT}"
bash scripts/rivanna/setup_cluster.sh

# Link latest L2 checkpoint if needed
if [[ -x scripts/rivanna/link_latest_checkpoint.sh ]]; then
  bash scripts/rivanna/link_latest_checkpoint.sh || true
fi

bash scripts/rivanna/submit_fullscale_variant_b.sh
