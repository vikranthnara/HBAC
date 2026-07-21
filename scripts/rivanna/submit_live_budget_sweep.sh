#!/usr/bin/env bash
# Submit live budget fraction sweep (Discovery D1) with DPO v2 LoRA.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/rivanna/_ssh_env.sh
source "${SCRIPT_DIR}/_ssh_env.sh"

bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_budget_sweep.sh" | awk '{print $NF}')
echo "Submitted live budget sweep array: ${JOB}"
