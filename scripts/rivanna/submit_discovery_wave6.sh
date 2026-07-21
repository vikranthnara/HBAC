#!/usr/bin/env bash
# Full baseline comparison (uniform, CLEAR, ZEBRA, HBAC) at floors 600+400.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"
JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_all_baselines.sh" | awk '{print $NF}')
echo "Submitted all-baselines live eval (600+400, incl ZEBRA): ${JOB}"
