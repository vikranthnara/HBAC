#!/usr/bin/env bash
# Submit SWE prompt-fix capability re-pilot.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

for f in \
  hbac/baselines/react.py \
  hbac/envs/swe_bench.py \
  hbac/envs/swe_local.py \
  hbac/envs/swe_registry.py \
  slurm/eval_capability_pilot_2xa100_vllm_swe_prompt.sh \
  paper/main.tex \
  "research docs/Research Discovery.md"; do
  "${RSYNC[@]}" "${LOCAL_ROOT}/${f}" "${RIVANNA_HOST}:${REMOTE_ROOT}/${f}"
done

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_capability_pilot_2xa100_vllm_swe_prompt.sh")

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
p.write_text(json.dumps({
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "vllm_swe_prompt_job": "${JOB}",
    "previous": {
        "vllm_swe_fix_job": prev.get("vllm_swe_fix_job", "17162445"),
        "vllm_job": prev.get("vllm_job", "17142925"),
    },
    "path_b_locked": True,
    "paper_primary": "holdout_+0.45pp",
    "model": "Qwen/Qwen3-Coder-Next-FP8",
    "fix": "SWE ReAct prompt + seeded query snippets",
    "output": "results/capability_pilot_vllm_swe_prompt_analysis.json",
    "status": "SUBMITTED",
}, indent=2) + "\n")
print(f"Wrote {p}")
PY

echo "CAPABILITY_SWE_PROMPT_JOB=${JOB}"
rivanna_ssh "squeue -u \$(whoami) -o '%.18i %.12P %.20j %.2t %.10M %R' | head -8"
