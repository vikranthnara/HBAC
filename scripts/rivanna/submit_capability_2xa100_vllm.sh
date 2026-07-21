#!/usr/bin/env bash
# Submit vLLM capability pilot (Qwen3-Coder-Next-FP8, 2× A100).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

for f in \
  hbac/core/llm.py \
  slurm/eval_capability_pilot_2xa100_vllm.sh; do
  "${RSYNC[@]}" "${LOCAL_ROOT}/${f}" "${RIVANNA_HOST}:${REMOTE_ROOT}/${f}"
done

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_capability_pilot_2xa100_vllm.sh")

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("${LOCAL_ROOT}/results/rivanna/capability_2xa100_jobs.json")
prev = {}
if p.is_file():
    try:
        prev = json.loads(p.read_text())
    except Exception:
        prev = {}
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "vllm_job": "${JOB}",
    "previous": {
        "gpu_only_job": prev.get("gpu_only_job"),
        "cpu_offload_job": prev.get("cpu_offload_job"),
        "previous_failed_job": prev.get("previous_failed_job", "17063216"),
    },
    "model": "Qwen/Qwen3-Coder-Next-FP8",
    "gpus": "a100:2",
    "mode": "vllm",
    "slurm": "slurm/eval_capability_pilot_2xa100_vllm.sh",
    "output": "results/capability_pilot_2xa100_vllm_analysis.json",
    "status": "SUBMITTED",
}, indent=2) + "\n")
print(f"Wrote {p}")
PY

echo "CAPABILITY_VLLM_JOB=${JOB}"
rivanna_ssh "squeue -u \$(whoami) -o '%.18i %.12P %.20j %.2t %.10M %R %S' | head -10"
