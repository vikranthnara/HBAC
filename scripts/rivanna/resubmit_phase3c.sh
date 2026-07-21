#!/usr/bin/env bash
# Resubmit Phase 3c after fixes: DPO v1 live eval + DPO v2 train + live eval chain.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/rivanna/_ssh_env.sh
source "${SCRIPT_DIR}/_ssh_env.sh"

echo "Syncing to Rivanna..."
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

# Cancel stale pending live eval from failed v2 train
rivanna_ssh "scancel 16764443 2>/dev/null || true"

V1_EVAL=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_compose_live_dpo.sh" | awk '{print $NF}')
echo "Submitted DPO v1 live eval (retry): ${V1_EVAL}"

V2_TRAIN=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/train_llm_dpo_v2.sh" | awk '{print $NF}')
echo "Submitted DPO v2 train (retry): ${V2_TRAIN}"

V2_EVAL=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --dependency=afterok:${V2_TRAIN} slurm/eval_compose_live_dpo_v2.sh" | awk '{print $NF}')
echo "Submitted DPO v2 live eval (after train): ${V2_EVAL}"
