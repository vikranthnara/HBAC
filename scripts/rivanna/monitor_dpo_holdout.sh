#!/usr/bin/env bash
# Monitor DPO holdout retrain (job 16854770); pull LoRA + audit on completion.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

DPO_JOB="${DPO_HOLDOUT_JOB:-${MONITOR_JOBS:-16854770}}"
POLL_SEC="${POLL_SEC:-120}"
LOG="${MONITOR_LOG:-${LOCAL_ROOT}/logs/monitor_dpo_holdout.log}"
TRACKER="${LOCAL_ROOT}/results/rivanna/dpo_holdout_jobs.json"
mkdir -p "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results/rivanna" "${LOCAL_ROOT}/checkpoints/llm_dpo"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"; }

job_state() {
  [[ -z "${DPO_JOB}" ]] && return 0
  rivanna_ssh "sacct -j ${DPO_JOB} --format=JobID,JobName%18,State,ExitCode,Elapsed,Start -n -P 2>/dev/null | grep -v batch | grep -v '\.batch'" || true
}

queue_state() {
  rivanna_ssh "squeue -u \$(whoami) -j ${DPO_JOB} 2>/dev/null" || true
}

all_done() {
  local states
  states="$(job_state)"
  [[ -z "${states}" ]] && return 1
  echo "${states}" | grep -qE 'RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED' && return 1
  return 0
}

job_failed() {
  job_state | grep -qE 'FAILED|CANCELLED|TIMEOUT|NODE_FAIL'
}

update_tracker() {
  local status="$1"
  python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("${TRACKER}")
data = json.loads(p.read_text()) if p.is_file() else {}
data.update({
    "dpo_holdout_job": "${DPO_JOB}",
    "monitor_script": "scripts/rivanna/monitor_dpo_holdout.sh",
    "monitor_log": "logs/monitor_dpo_holdout.log",
    "status": "${status}",
    "last_poll_at": datetime.now(timezone.utc).isoformat(),
})
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(data, indent=2) + "\\n")
PY
}

pull_artifacts() {
  log "Pulling DPO holdout artifacts..."
  "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/results/dpo_contamination_audit_holdout.json" \
    "${LOCAL_ROOT}/results/dpo_contamination_audit_holdout.json" 2>/dev/null || true
  "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/checkpoints/llm_dpo/*_capability_holdout/" \
    "${LOCAL_ROOT}/checkpoints/llm_dpo/" 2>/dev/null || true
  rivanna_ssh "ls -td ${REMOTE_ROOT}/checkpoints/llm_dpo/*_capability_holdout 2>/dev/null | head -1" \
    > "${LOCAL_ROOT}/results/rivanna/dpo_holdout_remote_ckpt.txt" 2>/dev/null || true
  rivanna_ssh "squeue -u \$(whoami)" | tee "${LOCAL_ROOT}/results/rivanna/rivanna_status.txt" || true
}

analyze_local() {
  cd "${LOCAL_ROOT}"
  if [[ -f results/dpo_contamination_audit_holdout.json ]]; then
    log "Holdout contamination audit:"
    python3 -c "import json; d=json.load(open('results/dpo_contamination_audit_holdout.json')); print('verdict=', d.get('verdict'), 'overlap=', d.get('overlap_count'))" \
      | tee -a "${LOG}" || true
  fi
  python -m hbac.scripts.lock_canonical_artifacts >> "${LOG}" 2>&1 || true
}

log "DPO_HOLDOUT_MONITOR_START job=${DPO_JOB} poll=${POLL_SEC}s"
update_tracker "MONITORING"

while true; do
  states="$(job_state)"
  queue="$(queue_state)"
  log "sacct:\n${states:-<no sacct>}"
  log "squeue:\n${queue:-<not in queue>}"
  pull_artifacts

  if [[ -n "${DPO_JOB}" ]] && all_done; then
    pull_artifacts
    analyze_local
    if job_failed; then
      log "DPO holdout job FAILED — check slurm logs on Rivanna"
      update_tracker "FAILED"
      log "MONITOR_DPO_HOLDOUT_FAILED"
      exit 1
    fi
    log "DPO holdout job COMPLETED"
    update_tracker "COMPLETED"
    log "MONITOR_DPO_HOLDOUT_COMPLETE"
    break
  fi

  if [[ -z "${DPO_JOB}" ]]; then
    log "No job ID; single pull pass"
    pull_artifacts
    analyze_local
    break
  fi

  update_tracker "MONITORING"
  sleep "${POLL_SEC}"
done
