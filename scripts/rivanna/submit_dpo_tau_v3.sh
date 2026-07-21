#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"
JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/train_llm_dpo_tau.sh" | awk '{print $NF}')
echo "Submitted DPO tau v3: ${JOB}"
