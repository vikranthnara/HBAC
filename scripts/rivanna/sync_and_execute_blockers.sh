#!/usr/bin/env bash
# Sync latest scripts to Rivanna run dir, execute blockers, pull results.
# Reads SSH_KEY_USER / SSH_KEY_PASSWORD from .env
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/rivanna/_ssh_env.sh
source "${SCRIPT_DIR}/_ssh_env.sh"

LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ARCHIVE="/tmp/hbac_blockers_sync.tar.gz"

echo "=== Rivanna: sync + execute blockers ==="
echo "Host: ${RIVANNA_HOST}"
echo "Root: ${REMOTE_ROOT}"

echo "[1/4] Testing SSH..."
rivanna_ssh "hostname && whoami"

echo "[2/4] Packaging scripts..."
tar czf "${ARCHIVE}" \
  --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude '.env' \
  -C "${LOCAL_ROOT}" \
  hbac scripts slurm pyproject.toml README.md

echo "[3/4] Upload + extract on cluster..."
rivanna_scp "${ARCHIVE}" "${RIVANNA_HOST}:${REMOTE_ROOT}/hbac_sync.tar.gz"
rivanna_ssh bash -s <<REMOTE
set -euo pipefail
cd '${REMOTE_ROOT}'
tar xzf hbac_sync.tar.gz
chmod +x scripts/rivanna/*.sh slurm/*.sh 2>/dev/null || true
bash scripts/rivanna/execute_blockers.sh
REMOTE

echo "[4/4] Pull results..."
bash "${SCRIPT_DIR}/pull_from_rivanna.sh"

echo "Done. Run: bash scripts/run_impact_loop.sh"
