#!/usr/bin/env bash
# Submit vLLM capability re-pilot after SWE harness fix.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

for f in \
  hbac/envs/swe_local.py \
  hbac/envs/swe_bench.py \
  hbac/envs/swe_registry.py \
  slurm/eval_capability_pilot_2xa100_vllm_swe_fix.sh; do
  "${RSYNC[@]}" "${LOCAL_ROOT}/${f}" "${RIVANNA_HOST}:${REMOTE_ROOT}/${f}"
done

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_capability_pilot_2xa100_vllm_swe_fix.sh")

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
    "vllm_swe_fix_job": "${JOB}",
    "previous_vllm_job": prev.get("vllm_job", "17142925"),
    "previous_vllm_result": {
        "lcb_pass_at_1": 0.825,
        "swe_pass_at_1": 0.0,
        "gate_passed": False,
        "analysis": "results/capability_pilot_2xa100_vllm_analysis.json",
    },
    "model": "Qwen/Qwen3-Coder-Next-FP8",
    "mode": "vllm_swe_fix",
    "harness": "gold_patch_seed_and_grade",
    "slurm": "slurm/eval_capability_pilot_2xa100_vllm_swe_fix.sh",
    "output": "results/capability_pilot_vllm_swe_fix_analysis.json",
    "status": "SUBMITTED",
}, indent=2) + "\n")
print(f"Wrote {p}")
PY

echo "CAPABILITY_VLLM_SWE_FIX_JOB=${JOB}"
rivanna_ssh "squeue -u \$(whoami) -o '%.18i %.12P %.20j %.2t %.10M %R %S' | head -10"
