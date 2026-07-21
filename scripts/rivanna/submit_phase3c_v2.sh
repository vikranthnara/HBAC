#!/usr/bin/env bash
# Submit Phase 3c v2: SFT warmstart + wrong_tool DPO + format gate + live eval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/rivanna/_ssh_env.sh
source "${SCRIPT_DIR}/_ssh_env.sh"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Syncing to Rivanna..."
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/train_llm_dpo_v2.sh" | awk '{print $NF}')
echo "Submitted DPO v2 train: ${JOB}"

EVAL_JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --dependency=afterok:${JOB} slurm/eval_compose_live_dpo_v2.sh" | awk '{print $NF}' || true)
if [[ -n "${EVAL_JOB:-}" ]]; then
  echo "Submitted live eval DPO v2 (after train): ${EVAL_JOB}"
fi
