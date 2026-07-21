#!/usr/bin/env bash
# Discovery wave 3: ControllerRunner live ablation (D6).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_controller_runner.sh" | awk '{print $NF}')
echo "Submitted ControllerRunner live eval (D6): ${JOB}"
