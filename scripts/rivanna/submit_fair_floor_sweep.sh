#!/usr/bin/env bash
# Submit hbac_fair vs type_prior floor dose-response (n~300, 6 floors × 2 allocators).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

rivanna_ssh "cd ${REMOTE_ROOT} && rm -rf results/fair_floor_sweep_shards && mkdir -p results/fair_floor_sweep_shards"

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_live_fair_floor_sweep_array.sh")
echo "Fair floor sweep (12 shards): ${JOB}"
rivanna_ssh "squeue -u \$(whoami) | head -15"
