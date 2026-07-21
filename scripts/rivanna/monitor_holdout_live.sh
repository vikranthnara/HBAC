#!/usr/bin/env bash
# Monitor holdout-LoRA V3 live array + merge; pull artifacts on completion.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

TRACKER="${LOCAL_ROOT}/results/rivanna/holdout_live_jobs.json"
if [[ -f "${TRACKER}" ]]; then
  ARRAY_JOB="$(python3 -c "import json; print(json.load(open('${TRACKER}')).get('array_job',''))")"
  MERGE_JOB="$(python3 -c "import json; print(json.load(open('${TRACKER}')).get('merge_job',''))")"
fi
ARRAY_JOB="${HOLDOUT_LIVE_ARRAY_JOB:-${ARRAY_JOB:-}}"
MERGE_JOB="${HOLDOUT_LIVE_MERGE_JOB:-${MERGE_JOB:-}}"
JOBS="${MONITOR_JOBS:-${ARRAY_JOB},${MERGE_JOB}}"
JOBS="$(echo "${JOBS}" | sed 's/^,//;s/,$//;s/,,/,/g')"
POLL_SEC="${POLL_SEC:-120}"
LOG="${MONITOR_LOG:-${LOCAL_ROOT}/logs/monitor_holdout_live.log}"
mkdir -p "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results/rivanna"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"; }

job_state() {
  [[ -z "${JOBS}" ]] && return 0
  rivanna_ssh "sacct -j ${JOBS} --format=JobID,JobName%18,State,ExitCode,Elapsed -n -P 2>/dev/null | grep -vE 'batch|\.batch|extern'" || true
}

all_done() {
  local states
  states="$(job_state)"
  [[ -z "${states}" ]] && return 1
  echo "${states}" | grep -qE 'RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED' && return 1
  return 0
}

update_tracker() {
  local status="$1"
  python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("${TRACKER}")
d = json.loads(p.read_text()) if p.is_file() else {}
d["status"] = "${status}"
d["last_poll_at"] = datetime.now(timezone.utc).isoformat()
d["array_job"] = "${ARRAY_JOB}"
d["merge_job"] = "${MERGE_JOB}"
p.write_text(json.dumps(d, indent=2) + "\n")
PY
}

pull_artifacts() {
  log "Pulling holdout live artifacts..."
  for f in \
    results/rivanna/compose_live_v3_holdout_floor400_n2000.json \
    results/paired_allocator_analysis_v3_holdout.json \
    results/v3_holdout_live_analysis.json; do
    "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/${f}" "${LOCAL_ROOT}/${f}" 2>/dev/null || true
  done
  "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/results/rivanna/v3_live_holdout_shards/" \
    "${LOCAL_ROOT}/results/rivanna/v3_live_holdout_shards/" 2>/dev/null || true
}

analyze_local() {
  cd "${LOCAL_ROOT}"
  if [[ -f results/rivanna/compose_live_v3_holdout_floor400_n2000.json ]]; then
    python -m hbac.scripts.analyze_v3_d18_live \
      --result-path results/rivanna/compose_live_v3_holdout_floor400_n2000.json \
      --paired-path results/paired_allocator_analysis_v3_holdout.json \
      --output results/v3_holdout_live_analysis.json >> "${LOG}" 2>&1 || true
    python -m hbac.scripts.lock_canonical_artifacts >> "${LOG}" 2>&1 || true
    log "HOLDOUT_LIVE_ANALYSIS_COMPLETE"
  fi
}

log "HOLDOUT_LIVE_MONITOR_START jobs=${JOBS} poll=${POLL_SEC}s"
update_tracker "MONITORING"

while true; do
  states="$(job_state)"
  log "State:\n${states:-<no jobs>}"
  pull_artifacts
  if [[ -n "${JOBS}" ]] && all_done; then
    pull_artifacts
    analyze_local
    update_tracker "COMPLETED"
    log "MONITOR_HOLDOUT_LIVE_COMPLETE"
    break
  fi
  if [[ -z "${JOBS}" ]]; then
    log "No job IDs; exiting"
    break
  fi
  sleep "${POLL_SEC}"
done
