#!/usr/bin/env bash
# Submit tau v3b live eval chained after DPO tau v3b training.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

DPO_JOB="${1:-}"
if [[ -z "${DPO_JOB}" ]]; then
  echo "Usage: $0 <dpo_tau_v3b_job_id>"
  echo "  Or:  DPO_JOB=16769472 $0"
  exit 1
fi

bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

TAU_V3B=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --dependency=afterok:${DPO_JOB} slurm/eval_live_tau_v3b.sh" | awk '{print $NF}')
echo "Submitted tau v3b live eval (afterok:${DPO_JOB}): ${TAU_V3B}"
