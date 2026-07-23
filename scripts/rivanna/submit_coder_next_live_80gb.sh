#!/usr/bin/env bash
# Retry Coder-Next live slice on A100-80GB only.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

for f in \
  slurm/eval_coder_next_live_slice.sh \
  "research docs/Path B Freeze.md" \
  "research docs/Research Discovery.md" \
  "research docs/Results.md"; do
  "${RSYNC[@]}" "${LOCAL_ROOT}/${f}" "${RIVANNA_HOST}:${REMOTE_ROOT}/${f}"
done

JOB=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_coder_next_live_slice.sh")

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
    "coder_next_live_slice_80gb_job": "${JOB}",
    "previous_live_slice_failed": "17192332",
    "fail_reason": "landed on A100-40GB; vLLM engine init failed",
    "constraint": "a100_80gb",
    "swe_fuzzy_final": {
        "job": "17192331",
        "lcb_pass_at_1": 0.825,
        "swe_pass_at_1": 0.0,
        "gate_passed": False,
        "verdict": "CLOSE_SWE_SALVAGE_PATH_B",
    },
    "path_b_freeze": True,
    "output": "results/paired_allocator_analysis_coder_next_slice.json",
    "status": "SUBMITTED",
    "previous": prev,
}, indent=2) + "\n")
print(f"Wrote {p}")
PY

echo "CODER_NEXT_LIVE_80GB_JOB=${JOB}"
rivanna_ssh "squeue -u \$(whoami) -o '%.18i %.12P %.22j %.2t %.10M %R' | head -8"
