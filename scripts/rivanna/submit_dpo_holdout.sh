#!/usr/bin/env bash
# Submit DPO holdout retrain (exclude LCB from training pairs).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/train_llm_dpo_holdout.sh")
echo "DPO_HOLDOUT_JOB=${JOB}"
rivanna_ssh "squeue -u \$(whoami) | head -10"
