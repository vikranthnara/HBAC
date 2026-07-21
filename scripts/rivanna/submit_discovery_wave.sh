#!/usr/bin/env bash
# Submit discovery follow-ups: tau-only live, tau-overweighted DPO v3b.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

TAU_LIVE=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_tau_only.sh" | awk '{print $NF}')
echo "Submitted tau-only live eval: ${TAU_LIVE}"

DPO=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/train_llm_dpo_tau_v3b.sh" | awk '{print $NF}')
echo "Submitted DPO tau v3b: ${DPO}"
