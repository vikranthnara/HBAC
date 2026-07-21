#!/usr/bin/env bash
# Monitor 2× A100 capability pilot; pull analysis on completion.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

TRACKER="${LOCAL_ROOT}/results/rivanna/capability_2xa100_jobs.json"
JOB="${CAPABILITY_2XA100_JOB:-}"
if [[ -z "${JOB}" && -f "${TRACKER}" ]]; then
  JOB="$(python3 -c "import json; print(json.load(open('${TRACKER}')).get('job',''))")"
fi
POLL_SEC="${POLL_SEC:-120}"
LOG="${MONITOR_LOG:-${LOCAL_ROOT}/logs/monitor_capability_2xa100.log}"
mkdir -p "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results/rivanna"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"; }

job_state() {
  rivanna_ssh "sacct -j ${JOB} --format=JobID,JobName%18,State,ExitCode,Elapsed -n -P 2>/dev/null | grep -vE 'batch|\.batch|extern'" || true
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
d["job"] = "${JOB}"
p.write_text(json.dumps(d, indent=2) + "\n")
PY
}

pull_artifacts() {
  log "Pulling capability 2xA100 artifacts..."
  for f in \
    results/capability_pilot_2xa100_analysis.json \
    results/rivanna/capability_pilot_2xa100_uniform.json; do
    "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/${f}" "${LOCAL_ROOT}/${f}" 2>/dev/null || true
  done
  "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/results/rivanna/capability_pilot_2xa100_shards/" \
    "${LOCAL_ROOT}/results/rivanna/capability_pilot_2xa100_shards/" 2>/dev/null || true
}

analyze_local() {
  cd "${LOCAL_ROOT}"
  if [[ -f results/rivanna/capability_pilot_2xa100_shards/uniform.json ]]; then
    python -m hbac.scripts.analyze_capability_pilot \
      --source results/rivanna/capability_pilot_2xa100_shards/uniform.json \
      --output results/capability_pilot_2xa100_analysis.json >> "${LOG}" 2>&1 || true
  fi
  if [[ -f results/capability_pilot_2xa100_analysis.json ]]; then
    python3 -c "import json; d=json.load(open('results/capability_pilot_2xa100_analysis.json')); print('gate_passed=', d.get('gate_passed'), 'swe=', d.get('swe_pass_at_1'), 'lcb=', d.get('lcb_pass_at_1'), 'verdict=', d.get('verdict'))" \
      | tee -a "${LOG}" || true
  fi
}

log "CAPABILITY_2XA100_MONITOR_START job=${JOB} poll=${POLL_SEC}s"
update_tracker "MONITORING"

while true; do
  states="$(job_state)"
  log "State:\n${states:-<no jobs>}"
  if [[ -n "${JOB}" ]] && all_done; then
    pull_artifacts
    analyze_local
    if echo "${states}" | grep -q FAILED; then
      log "Capability 2xA100 FAILED — check slurm-${JOB}.out"
      update_tracker "FAILED"
      rivanna_ssh "tail -60 ${REMOTE_ROOT}/slurm-${JOB}.out" | tee -a "${LOG}" || true
      exit 1
    fi
    update_tracker "COMPLETED"
    log "MONITOR_CAPABILITY_2XA100_COMPLETE"
    break
  fi
  sleep "${POLL_SEC}"
done
