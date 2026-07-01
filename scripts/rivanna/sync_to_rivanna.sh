#!/usr/bin/env bash
# Sync HBAC repo + data from local machine to Rivanna.
# Usage (from laptop): bash scripts/rivanna/sync_to_rivanna.sh
set -euo pipefail

RIVANNA_HOST="${RIVANNA_HOST:-eyu8ps@login.hpc.virginia.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/standard/liverobotics/hbac}"
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Syncing ${LOCAL_ROOT} -> ${RIVANNA_HOST}:${REMOTE_ROOT}"

rsync -avz --progress \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude '.env' \
  "${LOCAL_ROOT}/" "${RIVANNA_HOST}:${REMOTE_ROOT}/"

rsync -avz --progress \
  "${LOCAL_ROOT}/data/oracles/" "${RIVANNA_HOST}:${REMOTE_ROOT}/data/oracles/"

rsync -avz --progress \
  "${LOCAL_ROOT}/checkpoints/variant_a/" "${RIVANNA_HOST}:${REMOTE_ROOT}/checkpoints/variant_a/"

echo "Done. SSH: ssh -Y ${RIVANNA_HOST}"
