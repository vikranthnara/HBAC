#!/usr/bin/env bash
# After V3 oracle: submit D18 fairness retrain + live pilot (GPU backfill).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

# Cancel duplicate full n2000 job if pilot preferred (keep 16794000 for full run)
# rivanna_ssh "scancel 16793903 2>/dev/null" || true

D18=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/train_l1_fairness.sh" | awk '{print $NF}')
echo "D18 fairness L1 retrain: ${D18}"

PILOT=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_v3_pilot.sh" | awk '{print $NF}')
echo "V3 live pilot (n~300): ${PILOT}"

echo "Full n2000 job 16794000 remains queued for definitive live comparison."
