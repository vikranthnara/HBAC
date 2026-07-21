#!/usr/bin/env bash
# Pull Rivanna results + checkpoints to local results/rivanna/.
# Usage: bash scripts/rivanna/pull_from_rivanna.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/rivanna/_ssh_env.sh
source "${SCRIPT_DIR}/_ssh_env.sh"

LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST="${LOCAL_ROOT}/results/rivanna"

mkdir -p "${DEST}"

echo "Pulling ${RIVANNA_HOST}:${REMOTE_ROOT}/results/ -> ${DEST}/"

# High-priority impact-loop artifacts
FILES=(
  "compose_live_bf040_seed47_retrain.json"
  "compose_live_bf040_seed47_grpo_lora.json"
  "compose_live_bf040_seed47_v2_sft_grpo.json"
  "compose_live_bf040_seed47_v2_sft_only.json"
  "grpo_format_sft_grpo.json"
  "grpo_format_sft_only.json"
  "grpo_format_dpo.json"
  "grpo_format_dpo_v2.json"
  "compose_live_bf040_seed47_v2_dpo.json"
  "compose_live_bf040_seed47_dpo_v2.json"
  "compose_live_bf040_controller_runner_dpo_v2.json"
  "compose_live_bf040_floor400_dpo_v2.json"
  "compose_live_bf040_floor300_dpo_v2.json"
  "compose_live_bf040_floor450_dpo_v2.json"
  "compose_live_bf040_floor500_dpo_v2.json"
  "compose_live_bf040_floor400_all_baselines_dpo_v2.json"
  "compose_live_bf040_floor600_all_baselines_dpo_v2.json"
  "compose_live_bf040_floor400_scarcity_boost_dpo_v2.json"
  "compose_live_bf040_floor300_roi_skip_dpo_v2.json"
  "compose_live_bf040_floor400_scarcity_parse_penalty_dpo_v2.json"
  "compose_live_bf040_floor400_scarcity_refined_dpo_v2.json"
  "compose_live_v3_heuristics_floor400_n1000_dpo_v2.json"
  "v3_real_oracle_matrix.json"
  "v3_live_analysis.json"
  "compose_live_v3_pilot_floor400_dpo_v2.json"
  "d16_oracle_compare.json"
  "compose_live_bf020_floor400_dpo_v2.json"
  "compose_live_tau_only_bf040_dpo_v2.json"
  "compose_live_tau_only_bf040_dpo_tau_v3b.json"
  "grpo_format_dpo_tau_v3b.json"
  "compose_live_bf025_dpo_v2_sweep.json"
  "compose_live_bf030_dpo_v2_sweep.json"
  "compose_live_bf035_dpo_v2_sweep.json"
  "compose_live_bf040_dpo_v2_sweep.json"
  "capability_report.json"
  "h6_long_summary.json"
  "h6_long_with_credit.json"
  "h6_long_no_credit.json"
)

for f in "${FILES[@]}"; do
  "${RSYNC[@]}" \
    "${RIVANNA_HOST}:${REMOTE_ROOT}/results/${f}" \
    "${DEST}/" 2>/dev/null || echo "  (skip ${f} — not on cluster yet)"
done

# TRL GRPO checkpoint (080820Z)
"${RSYNC[@]}" \
  "${RIVANNA_HOST}:${REMOTE_ROOT}/checkpoints/llm_grpo/20260703T080820Z/" \
  "${LOCAL_ROOT}/checkpoints/llm_grpo/20260703T080820Z/" 2>/dev/null \
  || echo "  (skip llm_grpo/20260703T080820Z — not on cluster yet)"

# Job status snapshot
rivanna_ssh "squeue -u \$(whoami); ls -la ${REMOTE_ROOT}/results/compose_live_*retrain* ${REMOTE_ROOT}/logs/impact_wave_jobids.txt ${REMOTE_ROOT}/logs/blockers_executed.txt 2>/dev/null" \
  | tee "${DEST}/rivanna_status.txt" || true

echo "Done. Re-run: bash scripts/run_impact_loop.sh"
