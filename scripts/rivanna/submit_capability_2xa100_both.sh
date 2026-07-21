#!/usr/bin/env bash
# Submit both capability load strategies: gpu_only + cpu_offload.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

for f in \
  hbac/core/llm.py \
  slurm/eval_capability_pilot_2xa100_gpuonly.sh \
  slurm/eval_capability_pilot_2xa100_offload.sh; do
  "${RSYNC[@]}" "${LOCAL_ROOT}/${f}" "${RIVANNA_HOST}:${REMOTE_ROOT}/${f}"
done

JOB_GPU=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_capability_pilot_2xa100_gpuonly.sh")
JOB_OFFLOAD=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_capability_pilot_2xa100_offload.sh")

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("${LOCAL_ROOT}/results/rivanna/capability_2xa100_jobs.json")
p.write_text(json.dumps({
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "gpu_only_job": "${JOB_GPU}",
    "cpu_offload_job": "${JOB_OFFLOAD}",
    "previous_failed_job": "17063216",
    "model": "Qwen/Qwen3-Coder-Next",
    "gpus": "a100:2",
    "modes": {
        "gpu_only": {
            "job": "${JOB_GPU}",
            "env": "HBAC_BNB_GPU_ONLY=1",
            "slurm": "slurm/eval_capability_pilot_2xa100_gpuonly.sh",
            "output": "results/capability_pilot_2xa100_gpuonly_analysis.json",
        },
        "cpu_offload": {
            "job": "${JOB_OFFLOAD}",
            "env": "HBAC_BNB_CPU_OFFLOAD=1",
            "slurm": "slurm/eval_capability_pilot_2xa100_offload.sh",
            "output": "results/capability_pilot_2xa100_offload_analysis.json",
        },
    },
    "status": "SUBMITTED",
}, indent=2) + "\n")
print(f"Wrote {p}")
PY

echo "CAPABILITY_GPU_ONLY_JOB=${JOB_GPU}"
echo "CAPABILITY_CPU_OFFLOAD_JOB=${JOB_OFFLOAD}"
rivanna_ssh "squeue -u \$(whoami) -o '%.18i %.12P %.20j %.2t %.10M %R %S' | head -10"
