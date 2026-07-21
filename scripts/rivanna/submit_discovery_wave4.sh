#!/usr/bin/env bash
# Discovery wave 4: floor sweep + extreme scarcity (D8/D9).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

FLOOR=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_floor_sweep.sh" | awk '{print $NF}')
echo "Submitted live floor sweep (400/500): ${FLOOR}"

BF020=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_extreme_scarcity.sh" | awk '{print $NF}')
echo "Submitted extreme scarcity bf=0.20 floor=400: ${BF020}"
