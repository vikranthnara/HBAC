#!/usr/bin/env bash
# Monitor capability pilot + V3 d18 live array; pull artifacts and run analysis on completion.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

CAP_JOB="${CAPABILITY_PILOT_JOB:-}"
ARRAY_JOB="${V3_LIVE_ARRAY_JOB:-}"
MERGE_JOB="${V3_LIVE_MERGE_JOB:-}"
JOBS="${MONITOR_JOBS:-}"
if [[ -z "${JOBS}" ]]; then
  JOBS="$(printf '%s' "${CAP_JOB}" | sed '/^$/d')"
  [[ -n "${ARRAY_JOB}" ]] && JOBS="${JOBS},${ARRAY_JOB}"
  [[ -n "${MERGE_JOB}" ]] && JOBS="${JOBS},${MERGE_JOB}"
fi
POLL_SEC="${POLL_SEC:-90}"
LOG="${MONITOR_LOG:-${LOCAL_ROOT}/logs/monitor_v3_d18.log}"
mkdir -p "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results/rivanna"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"; }

job_state() {
  [[ -z "${JOBS}" ]] && return 0
  rivanna_ssh "sacct -j ${JOBS} --format=JobID,JobName%20,State,ExitCode,Elapsed -n -P 2>/dev/null | grep -v batch | grep -v '\.batch'" || true
}

all_done() {
  local states
  states="$(job_state)"
  [[ -z "${states}" ]] && return 1
  echo "${states}" | grep -qE 'RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED' && return 1
  return 0
}

pull_artifacts() {
  log "Pulling artifacts from ${REMOTE_ROOT}..."
  for f in \
    results/rivanna/capability_pilot_uniform.json \
    results/capability_pilot_analysis.json \
    results/rivanna/compose_live_v3_d18_floor400_n2000.json \
    results/paired_allocator_analysis_v3_d18.json; do
    "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/${f}" "${LOCAL_ROOT}/${f}" 2>/dev/null || true
  done
  "${RSYNC[@]}" "${RIVANNA_HOST}:${REMOTE_ROOT}/results/rivanna/v3_live_d18_shards/" \
    "${LOCAL_ROOT}/results/rivanna/v3_live_d18_shards/" 2>/dev/null || true
  rivanna_ssh "squeue -u \$(whoami)" | tee "${LOCAL_ROOT}/results/rivanna/rivanna_status.txt" || true
}

analyze_local() {
  cd "${LOCAL_ROOT}"
  if [[ -f results/rivanna/capability_pilot_uniform.json ]]; then
    log "Analyzing capability pilot..."
    python -m hbac.scripts.analyze_capability_pilot \
      --source results/rivanna/capability_pilot_uniform.json \
      --output results/capability_pilot_analysis.json >> "${LOG}" 2>&1 || true
  fi
  if [[ -f results/rivanna/compose_live_v3_d18_floor400_n2000.json ]]; then
    log "Running paired allocator analysis..."
    python -m hbac.scripts.analyze_paired_allocators \
      --shard-dir results/rivanna/v3_live_d18_shards \
      --merged results/rivanna/compose_live_v3_d18_floor400_n2000.json \
      --pairs "hbac_d18:type_prior,hbac_guardrail:type_prior,hbac_joint:type_prior" \
      --output results/paired_allocator_analysis_v3_d18.json >> "${LOG}" 2>&1 || true
    python -m hbac.scripts.migrate_n2000_artifact >> "${LOG}" 2>&1 || true
    python -m hbac.scripts.lock_canonical_artifacts >> "${LOG}" 2>&1 || true
    log "V3_D18_ANALYSIS_COMPLETE"
  fi
}

log "Monitoring jobs: ${JOBS:-none} (poll=${POLL_SEC}s)"
while true; do
  states="$(job_state)"
  log "State:\n${states:-<no jobs>}"
  pull_artifacts
  analyze_local
  if [[ -n "${JOBS}" ]] && all_done; then
    log "All monitored jobs finished"
    pull_artifacts
    analyze_local
    break
  fi
  if [[ -z "${JOBS}" ]]; then
    log "No job IDs set; single pull pass"
    break
  fi
  sleep "${POLL_SEC}"
done
log "MONITOR_V3_D18_COMPLETE"
