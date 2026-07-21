#!/usr/bin/env bash
# Monitor V3 Rivanna jobs (16793999 oracle, 16794000 live n2000); pull + analyze on completion.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

JOBS="${MONITOR_JOBS:-16809196,16809201}"
POLL_SEC="${POLL_SEC:-120}"
LOG="${MONITOR_LOG:-${LOCAL_ROOT}/logs/monitor_v3.log}"
mkdir -p "${LOCAL_ROOT}/logs"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"; }

job_state() {
  rivanna_ssh "sacct -j ${JOBS} --format=JobID,JobName,State,ExitCode,Elapsed -n -P 2>/dev/null | grep -v batch" || true
}

all_done() {
  local states
  states="$(job_state)"
  [[ -z "${states}" ]] && return 1
  echo "${states}" | grep -qE 'RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED' && return 1
  return 0
}

pull_v3() {
  log "Pulling V3 artifacts..."
  "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/results/v3_real_oracle_matrix.json" \
    "${LOCAL_ROOT}/results/rivanna/" 2>/dev/null || true
  "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/results/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json" \
    "${LOCAL_ROOT}/results/rivanna/" 2>/dev/null || true
  "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/results/compose_live_v3_pilot_floor400_dpo_v2.json" \
    "${LOCAL_ROOT}/results/rivanna/" 2>/dev/null || true
  rivanna_ssh "squeue -u \$(whoami)" | tee "${LOCAL_ROOT}/results/rivanna/rivanna_status.txt" || true
}

analyze() {
  cd "${LOCAL_ROOT}"
  if [[ -f results/rivanna/v3_real_oracle_matrix.json ]]; then
    log "Oracle matrix on cluster pulled"
  fi
  if [[ -f results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json ]]; then
    SRC=results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json
  elif [[ -f results/rivanna/compose_live_v3_pilot_floor400_dpo_v2.json ]]; then
    SRC=results/rivanna/compose_live_v3_pilot_floor400_dpo_v2.json
  else
    SRC=""
  fi
  if [[ -n "${SRC}" ]]; then
    log "Running analyze_v3_live on ${SRC}..."
    python -m hbac.scripts.analyze_v3_live --result-path "${SRC}" --output results/v3_live_analysis.json >> "${LOG}" 2>&1
    python -m hbac.scripts.auto_update_research_docs >> "${LOG}" 2>&1 || true
    log "V3_LIVE_ANALYSIS_COMPLETE"
  else
    log "Live V3 result not yet available"
  fi
}

log "Monitoring jobs: ${JOBS} (poll=${POLL_SEC}s)"
while true; do
  states="$(job_state)"
  log "State:\n${states}"
  pull_v3
  if all_done; then
    log "All jobs finished"
    analyze
    break
  fi
  # Partial: analyze live if file exists (job may still be writing — check size stable)
  if [[ -f "${LOCAL_ROOT}/results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json" ]]; then
    sz=$(stat -f%z "${LOCAL_ROOT}/results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json" 2>/dev/null \
      || stat -c%s "${LOCAL_ROOT}/results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json" 2>/dev/null \
      || echo 0)
    log "Live result file present (${sz} bytes)"
  fi
  sleep "${POLL_SEC}"
done
log "MONITOR_V3_COMPLETE"
