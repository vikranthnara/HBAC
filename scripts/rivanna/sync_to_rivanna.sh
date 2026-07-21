#!/usr/bin/env bash
# Sync HBAC repo + data from local machine to Rivanna.
# Usage (from laptop): bash scripts/rivanna/sync_to_rivanna.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/rivanna/_ssh_env.sh
source "${SCRIPT_DIR}/_ssh_env.sh"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Syncing ${LOCAL_ROOT} -> ${RIVANNA_HOST}:${REMOTE_ROOT}"

"${RSYNC[@]}" \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude '.env' \
  "${LOCAL_ROOT}/" "${RIVANNA_HOST}:${REMOTE_ROOT}/"

"${RSYNC[@]}" \
  "${LOCAL_ROOT}/data/oracles/" "${RIVANNA_HOST}:${REMOTE_ROOT}/data/oracles/" 2>/dev/null || true

echo "Done. Remote: ${REMOTE_ROOT}"
