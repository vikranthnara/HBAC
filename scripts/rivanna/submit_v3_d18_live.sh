#!/usr/bin/env bash
# Submit V3 live d18 array + merge on Rivanna.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

echo "Syncing to ${REMOTE_ROOT}..."
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

rivanna_ssh "cd ${REMOTE_ROOT} && rm -rf results/rivanna/v3_live_d18_shards && mkdir -p results/rivanna/v3_live_d18_shards"

ARRAY_JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_live_v3_d18_array.sh")
MERGE_JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable --dependency=afterok:${ARRAY_JOB} slurm/merge_live_v3_d18.sh")

echo "V3_LIVE_ARRAY_JOB=${ARRAY_JOB}"
echo "V3_LIVE_MERGE_JOB=${MERGE_JOB}"
rivanna_ssh "squeue -u \$(whoami) | head -20"
