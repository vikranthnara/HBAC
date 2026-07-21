#!/usr/bin/env bash
# V3 wave: real oracle pool + live n1000 heuristic comparison on Rivanna.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

echo "=== Sync repo + oracles to Rivanna ==="
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

echo "=== Build real eval pool on cluster (LCB 500 + SWE Lite 50) ==="
rivanna_ssh "cd ${REMOTE_ROOT} && python -m hbac.scripts.build_real_eval_pool \
  --lcb-problems 500 --swe-limit 50 --output data/oracles/real_eval" || true

echo "=== Submit oracle matrix (CPU) ==="
ORACLE_JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_real_oracle_matrix.sh" | awk '{print $NF}')
echo "Oracle matrix job: ${ORACLE_JOB}"

echo "=== Submit live n1000 @ floor=400 (GPU) ==="
LIVE_JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_heuristics_n1000.sh" | awk '{print $NF}')
echo "Live n1000 job: ${LIVE_JOB}"

echo "Monitor: rivanna_ssh 'squeue -u \$(whoami)'"
echo "Pull: bash scripts/rivanna/pull_from_rivanna.sh"
