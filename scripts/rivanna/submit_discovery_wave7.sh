#!/usr/bin/env bash
# Wave 7: D12 scarcity boost + ensure P0-P1 jobs synced.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

SCARCITY=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_scarcity_boost.sh" | awk '{print $NF}')
echo "Submitted D12 scarcity boost live (floor=400): ${SCARCITY}"
