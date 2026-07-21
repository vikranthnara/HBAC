#!/usr/bin/env bash
# Discovery wave 5: floor dose-response (300, 450) — map pass@1 transition.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_floor_dose.sh" | awk '{print $NF}')
echo "Submitted floor dose-response (300/450): ${JOB}"
