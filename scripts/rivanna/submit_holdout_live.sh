#!/usr/bin/env bash
# Submit V3 D18 live eval using LCB-holdout DPO LoRA.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

echo "Syncing to ${REMOTE_ROOT}..."
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

# Confirm holdout LoRA exists remotely.
rivanna_ssh "ls -td ${REMOTE_ROOT}/checkpoints/llm_dpo/*_capability_holdout/model 2>/dev/null | head -1"

rivanna_ssh "cd ${REMOTE_ROOT} && rm -rf results/rivanna/v3_live_holdout_shards && mkdir -p results/rivanna/v3_live_holdout_shards"

ARRAY_JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_live_v3_holdout_array.sh")
MERGE_JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable --dependency=afterok:${ARRAY_JOB} slurm/merge_live_v3_holdout.sh")

TRACKER="${LOCAL_ROOT}/results/rivanna/holdout_live_jobs.json"
python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("${TRACKER}")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "array_job": "${ARRAY_JOB}",
    "merge_job": "${MERGE_JOB}",
    "lora": "checkpoints/llm_dpo/*_capability_holdout/model",
    "slurm_array": "slurm/eval_live_v3_holdout_array.sh",
    "slurm_merge": "slurm/merge_live_v3_holdout.sh",
    "allocators": ["hbac_d18", "type_prior", "hbac_joint", "hbac_guardrail", "uniform"],
    "output": "results/rivanna/compose_live_v3_holdout_floor400_n2000.json",
    "status": "SUBMITTED",
}, indent=2) + "\n")
print(f"Wrote {p}")
PY

echo "HOLDOUT_LIVE_ARRAY_JOB=${ARRAY_JOB}"
echo "HOLDOUT_LIVE_MERGE_JOB=${MERGE_JOB}"
rivanna_ssh "squeue -u \$(whoami) | head -20"
