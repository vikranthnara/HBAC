#!/usr/bin/env bash
# Submit Qwen3-Coder-Next 4-bit capability pilot on 2× A100 40GB.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

echo "Targeted sync of loader + slurm scripts..."
"${RSYNC[@]}" \
  hbac/core/llm.py \
  slurm/eval_capability_pilot_2xa100.sh \
  scripts/rivanna/submit_capability_2xa100.sh \
  scripts/rivanna/monitor_capability_2xa100.sh \
  "${RIVANNA_HOST}:${REMOTE_ROOT}/" 2>/dev/null || true

# Explicit paths (rsync above may flatten; sync with relative tree)
"${RSYNC[@]}" "${LOCAL_ROOT}/hbac/core/llm.py" \
  "${RIVANNA_HOST}:${REMOTE_ROOT}/hbac/core/llm.py"
"${RSYNC[@]}" "${LOCAL_ROOT}/slurm/eval_capability_pilot_2xa100.sh" \
  "${RIVANNA_HOST}:${REMOTE_ROOT}/slurm/eval_capability_pilot_2xa100.sh"

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_capability_pilot_2xa100.sh")
echo "CAPABILITY_2XA100_JOB=${JOB}"

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("${LOCAL_ROOT}/results/rivanna/capability_2xa100_jobs.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "job": "${JOB}",
    "model": "Qwen/Qwen3-Coder-Next",
    "gpus": "a100:2",
    "quant": "nf4_4bit",
    "slurm": "slurm/eval_capability_pilot_2xa100.sh",
    "output": "results/capability_pilot_2xa100_analysis.json",
    "status": "SUBMITTED",
}, indent=2) + "\n")
print(f"Wrote {p}")
PY

rivanna_ssh "squeue -u \$(whoami) -o '%.18i %.12P %.20j %.2t %.10M %R %S' | head -15"
