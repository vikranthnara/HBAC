#!/usr/bin/env bash
# Submit Path B follow-ups: SWE fuzzy salvage + Coder-Next live slice.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

for f in \
  hbac/envs/swe_local.py \
  hbac/envs/swe_bench.py \
  hbac/baselines/react.py \
  hbac/scripts/analyze_paired_allocators.py \
  slurm/eval_capability_pilot_2xa100_vllm_swe_fuzzy.sh \
  slurm/eval_coder_next_live_slice.sh \
  paper/main.tex \
  "research docs/Path B Freeze.md" \
  "research docs/Preregistered Analysis.md" \
  "research docs/Research Discovery.md" \
  "research docs/Results.md"; do
  "${RSYNC[@]}" "${LOCAL_ROOT}/${f}" "${RIVANNA_HOST}:${REMOTE_ROOT}/${f}"
done

JOB_FUZZY=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_capability_pilot_2xa100_vllm_swe_fuzzy.sh")
JOB_LIVE=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch --parsable slurm/eval_coder_next_live_slice.sh")

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
    "path_b_freeze": True,
    "swe_fuzzy_job": "${JOB_FUZZY}",
    "coder_next_live_slice_job": "${JOB_LIVE}",
    "previous_swe_prompt_job": prev.get("vllm_swe_prompt_job", "17187328"),
    "outputs": {
        "fuzzy": "results/capability_pilot_vllm_swe_fuzzy_analysis.json",
        "live_slice_paired": "results/paired_allocator_analysis_coder_next_slice.json",
    },
    "status": "SUBMITTED",
}, indent=2) + "\n")
print(f"Wrote {p}")
PY

echo "SWE_FUZZY_JOB=${JOB_FUZZY}"
echo "CODER_NEXT_LIVE_SLICE_JOB=${JOB_LIVE}"
rivanna_ssh "squeue -u \$(whoami) -o '%.18i %.12P %.22j %.2t %.10M %R' | head -10"
