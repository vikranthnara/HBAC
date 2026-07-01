#!/usr/bin/env bash
# Deploy HBAC to a NEW isolated folder on Rivanna and submit full-scale Variant B + LLM jobs.
# Run from your Mac (with UVA VPN if off-campus):
#   bash scripts/rivanna/deploy_new_run.sh
#
# Optional: SSHPASS='your-password' bash scripts/rivanna/deploy_new_run.sh
set -euo pipefail

RIVANNA_HOST="${RIVANNA_HOST:-eyu8ps@login.hpc.virginia.edu}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_DIR="${REMOTE_DIR:-/standard/liverobotics/hbac-run-${RUN_ID}}"
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARCHIVE="/tmp/hbac_deploy_${RUN_ID}.tar.gz"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)
if [[ -n "${SSHPASS:-}" ]]; then
  SSH_CMD=(sshpass -e ssh "${SSH_OPTS[@]}")
  SCP_CMD=(sshpass -e scp "${SSH_OPTS[@]}")
  export SSHPASS
else
  SSH_CMD=(ssh "${SSH_OPTS[@]}")
  SCP_CMD=(scp "${SSH_OPTS[@]}")
fi

echo "=== HBAC Rivanna full-scale deploy ==="
echo "Local:  ${LOCAL_ROOT}"
echo "Remote: ${REMOTE_DIR}"
echo "Host:   ${RIVANNA_HOST}"

# Package (exclude secrets)
echo "Packaging..."
tar czf "${ARCHIVE}" \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.git' --exclude '.env' \
  -C "${LOCAL_ROOT}" .

echo "Creating remote directory (does not touch other folders)..."
"${SSH_CMD[@]}" "${RIVANNA_HOST}" "mkdir -p '${REMOTE_DIR}' && echo OK: ${REMOTE_DIR}"

echo "Uploading (${ARCHIVE})..."
"${SCP_CMD[@]}" "${ARCHIVE}" "${RIVANNA_HOST}:${REMOTE_DIR}/hbac_deploy.tar.gz"

echo "Extracting and submitting jobs..."
"${SSH_CMD[@]}" "${RIVANNA_HOST}" bash -s <<REMOTE
set -euo pipefail
export HBAC_ROOT="${REMOTE_DIR}"
cd "${REMOTE_DIR}"
tar xzf hbac_deploy.tar.gz
export HBAC_ROOT="${REMOTE_DIR}"

module purge
module load miniforge/24.3.0-py3.11
if ! conda env list | grep -q '^hbac '; then
  conda create -n hbac python=3.11 -y
fi
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
pip install -q -e ".[dev]"

mkdir -p logs checkpoints results data/oracles
if [[ -d checkpoints/variant_a ]]; then
  bash scripts/rivanna/link_latest_checkpoint.sh || true
fi

# Full-scale Variant B: parallel Stage 3 array + Stage 4 joint
export SKIP_SMOKE=0
bash scripts/rivanna/submit_fullscale_variant_b.sh

# Phase 3b LLM GRPO on A100 (separate job; needs vLLM on compute node)
mkdir -p slurm logs
if [[ -f slurm/train_llm_grpo.sh ]]; then
  LLM_JOB=\$(sbatch --parsable slurm/train_llm_grpo.sh)
  echo "LLM GRPO job: \${LLM_JOB}"
  echo "llm_grpo=\${LLM_JOB}" >> logs/fullscale_variant_b_jobids.txt
fi

echo ""
echo "=== Deployed to ${REMOTE_DIR} ==="
echo "Monitor: squeue -u \$USER"
echo "Job IDs: cat ${REMOTE_DIR}/logs/fullscale_variant_b_jobids.txt"
REMOTE

echo ""
echo "Done. Remote path: ${REMOTE_DIR}"
echo "SSH: ssh -Y ${RIVANNA_HOST}"
echo "Then: cd ${REMOTE_DIR} && squeue -u \$USER"
