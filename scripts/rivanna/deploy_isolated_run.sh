#!/usr/bin/env bash
# Deploy HBAC to a NEW isolated folder on Rivanna and submit full-scale Variant B.
# Does NOT modify /standard/liverobotics/hbac or any existing paths.
#
# Prerequisites: UVA VPN, sshpass (brew install hudochenkov/sshpass/sshpass)
#
# Usage (from laptop, repo root):
#   export SSHPASS='your-rivanna-password'   # or omit and type when prompted
#   bash scripts/rivanna/deploy_isolated_run.sh
set -euo pipefail

RIVANNA_HOST="${RIVANNA_HOST:-eyu8ps@login.hpc.virginia.edu}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_DIR="${REMOTE_DIR:-/standard/liverobotics/hbac-run-${RUN_ID}}"
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARCHIVE="/tmp/hbac_deploy_${RUN_ID}.tar.gz"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null; then
  SSH=(sshpass -e ssh "${SSH_OPTS[@]}")
  SCP=(sshpass -e scp "${SSH_OPTS[@]}")
else
  SSH=(ssh "${SSH_OPTS[@]}")
  SCP=(scp "${SSH_OPTS[@]}")
fi

echo "=== HBAC Rivanna isolated deploy ==="
echo "Remote: ${REMOTE_DIR}"
echo "Host:   ${RIVANNA_HOST}"

echo "[1/5] Packaging..."
tar czf "${ARCHIVE}" \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.git' --exclude '.env' \
  -C "${LOCAL_ROOT}" .

echo "[2/5] Testing SSH..."
"${SSH[@]}" "${RIVANNA_HOST}" "hostname && whoami"

echo "[3/5] Creating isolated directory (no changes to sibling paths)..."
"${SSH[@]}" "${RIVANNA_HOST}" "mkdir -p '${REMOTE_DIR}' && ls -la /standard/liverobotics | tail -5"

echo "[4/5] Uploading (${ARCHIVE})..."
"${SCP[@]}" "${ARCHIVE}" "${RIVANNA_HOST}:${REMOTE_DIR}/hbac_deploy.tar.gz"

echo "[5/5] Extract, setup, submit..."
"${SSH[@]}" "${RIVANNA_HOST}" bash -s <<REMOTE
set -euo pipefail
cd '${REMOTE_DIR}'
tar xzf hbac_deploy.tar.gz
export HBAC_ROOT='${REMOTE_DIR}'
export HBAC_RUN_ID='${RUN_ID}'

module purge
module load miniforge/24.3.0-py3.11
if ! conda env list | grep -q '^hbac '; then
  conda create -n hbac python=3.11 -y
fi
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
pip install -q -e ".[dev]"

mkdir -p data/oracles checkpoints logs results
if [[ -d /standard/liverobotics/hbac/data/oracles ]]; then
  echo "Linking oracles from shared hbac (read-only)..."
  ln -sfn /standard/liverobotics/hbac/data/oracles data/oracles-shared 2>/dev/null || true
  if [[ ! -d data/oracles ]] || [[ -z "\$(ls -A data/oracles 2>/dev/null)" ]]; then
    cp -r /standard/liverobotics/hbac/data/oracles data/oracles 2>/dev/null || true
  fi
fi
if [[ -d /standard/liverobotics/hbac/checkpoints/variant_a ]]; then
  cp -r /standard/liverobotics/hbac/checkpoints/variant_a checkpoints/ 2>/dev/null || true
fi
bash scripts/rivanna/link_latest_checkpoint.sh 2>/dev/null || true

export SKIP_SMOKE=0
bash scripts/rivanna/submit_fullscale_variant_b.sh

echo ""
echo "=== Deployed to ${REMOTE_DIR} ==="
echo "Monitor: squeue -u \\\$USER"
echo "Job IDs:  cat ${REMOTE_DIR}/logs/fullscale_variant_b_jobids.txt"
REMOTE

echo ""
echo "Done. Isolated run at: ${REMOTE_DIR}"
echo "SSH: ssh -Y ${RIVANNA_HOST}"
echo "Then: cd ${REMOTE_DIR} && squeue -u \$USER"
