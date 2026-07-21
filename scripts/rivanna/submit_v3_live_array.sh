#!/usr/bin/env bash
# Cancel monolithic n2000 job; submit checkpointed array + merge dependency.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

OLD_JOB="${1:-16794000}"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

echo "Cancelling job ${OLD_JOB}..."
rivanna_ssh "scancel ${OLD_JOB} 2>/dev/null || true"

echo "Clearing stale shard dir (if any)..."
rivanna_ssh "cd ${REMOTE_ROOT} && rm -rf results/live_n2000_shards && mkdir -p results/live_n2000_shards"

ARRAY_JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_live_heuristics_n1000_array.sh")
echo "V3 live array (12 allocators): ${ARRAY_JOB}"

MERGE_JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable --dependency=afterok:${ARRAY_JOB} slurm/merge_live_n2000.sh")
echo "Merge job (after array): ${MERGE_JOB}"

rivanna_ssh "squeue -u \$(whoami) | head -20"
