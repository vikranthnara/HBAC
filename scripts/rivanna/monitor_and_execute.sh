#!/usr/bin/env bash
# Poll Rivanna jobs; on completion pull, analyze, update docs, optional follow-up submit.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

POLL_SEC="${POLL_SEC:-180}"
MAX_HOURS="${MAX_HOURS:-12}"
JOBS="${MONITOR_JOBS:?Set MONITOR_JOBS to comma-separated job IDs}"
LOG="${MONITOR_LOG:-${LOCAL_ROOT}/logs/monitor.log}"
FOLLOWUP_SCRIPT="${MONITOR_FOLLOWUP:-}"
mkdir -p "${LOCAL_ROOT}/logs"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"; }

rivanna_ssh_retry() {
  local n=0 max=5
  until rivanna_ssh "$@"; do
    n=$((n + 1))
    if [[ $n -ge $max ]]; then return 1; fi
    log "SSH retry ${n}/${max}..."
    sleep 30
  done
}

job_state() {
  rivanna_ssh_retry "sacct -j ${JOBS} --format=JobID,State,ExitCode -n -P 2>/dev/null | grep -v batch" || true
}

all_done() {
  local states
  states="$(job_state)"
  [[ -z "${states}" ]] && return 1
  echo "${states}" | grep -qE 'RUNNING|PENDING|CONFIGURING|COMPLETING' && return 1
  true
}

any_failed() {
  job_state | grep -q FAILED
}

pull_and_analyze() {
  log "Pulling results..."
  bash "${SCRIPT_DIR}/pull_from_rivanna.sh" >> "${LOG}" 2>&1 || log "WARN: pull failed"
  cd "${LOCAL_ROOT}"
  for mod in analyze_floor_sweep analyze_baseline_pareto analyze_compliant_utility analyze_unified_narrative; do
    log "Running hbac.scripts.${mod}..."
    python -m "hbac.scripts.${mod}" >> "${LOG}" 2>&1 || log "WARN: ${mod} failed"
  done
  log "Locking canonical artifacts..."
  python -m hbac.scripts.lock_canonical_artifacts >> "${LOG}" 2>&1 || log "WARN: canonical lock failed"
  log "Auto-updating research docs..."
  python -m hbac.scripts.auto_update_research_docs >> "${LOG}" 2>&1 || log "WARN: doc update failed"
}

run_followups() {
  if [[ -n "${FOLLOWUP_SCRIPT}" && -x "${SCRIPT_DIR}/${FOLLOWUP_SCRIPT}" ]]; then
    log "Running follow-up: ${FOLLOWUP_SCRIPT}"
    bash "${SCRIPT_DIR}/${FOLLOWUP_SCRIPT}" >> "${LOG}" 2>&1 || log "WARN: follow-up failed"
  fi
}

finalize() {
  pull_and_analyze
  run_followups
  python3 - <<'PY' >> "${LOG}" 2>&1
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("results/experiment_summary.json")
if p.is_file():
    d = json.loads(p.read_text())
    d["updated_at"] = datetime.now(timezone.utc).isoformat()
    d["monitor_status"] = "complete"
    d["monitor_jobs"] = __import__("os").environ.get("MONITOR_JOBS", "")
    p.write_text(json.dumps(d, indent=2) + "\n")
PY
  log "MONITOR_COMPLETE"
}

START=$(date +%s)
DEADLINE=$((START + MAX_HOURS * 3600))
LAST_PULL=""

log "MONITOR_START jobs=${JOBS} poll=${POLL_SEC}s log=${LOG}"

while [[ $(date +%s) -lt ${DEADLINE} ]]; do
  states="$(job_state)"
  log "=== poll ==="
  echo "${states}" | tee -a "${LOG}"

  if echo "${states}" | grep -q COMPLETED; then
    now=$(date +%s)
    if [[ -z "${LAST_PULL}" ]] || (( now - LAST_PULL > 300 )); then
      pull_and_analyze
      LAST_PULL=${now}
    fi
  fi

  if all_done; then
    log "ALL_JOBS_DONE"
    if any_failed; then
      log "WARN: one or more jobs FAILED"
    fi
    finalize
    exit 0
  fi

  sleep "${POLL_SEC}"
done

log "MONITOR_TIMEOUT after ${MAX_HOURS}h"
exit 1
