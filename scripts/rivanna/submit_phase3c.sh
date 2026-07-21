#!/usr/bin/env bash
# Submit Phase 3c: stub oracles + capability analysis + DPO train + format eval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/rivanna/_ssh_env.sh
source "${SCRIPT_DIR}/_ssh_env.sh"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Syncing to Rivanna..."
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/train_llm_dpo.sh" | awk '{print $NF}')
echo "Submitted DPO train: ${JOB}"

EVAL_JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --dependency=afterok:${JOB} slurm/eval_compose_live_dpo.sh" | awk '{print $NF}' || true)
if [[ -n "${EVAL_JOB:-}" ]]; then
  echo "Submitted live eval (after DPO): ${EVAL_JOB}"
fi
