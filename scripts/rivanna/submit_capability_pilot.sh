#!/usr/bin/env bash
# Submit capability pilot (uniform only, SWE+LCB gate) on Rivanna.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

echo "Syncing to ${REMOTE_ROOT}..."
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_capability_pilot.sh")
echo "CAPABILITY_PILOT_JOB=${JOB}"
rivanna_ssh "squeue -u \$(whoami) | head -10"
