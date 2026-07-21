#!/usr/bin/env bash
# Post-Wave-8 next steps: D12 refined live + D16 oracle compare on Rivanna.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

BASE_L1=$(rivanna_ssh "ls -td ${REMOTE_ROOT}/checkpoints/variant_b/parallel_tight/bf040_seed47/stage3/*/level1_policy.npz 2>/dev/null | head -1")
D16_L1=$(rivanna_ssh "ls -td ${REMOTE_ROOT}/checkpoints/phase3_parse_penalty_0.3/*/stage3/level1_policy.npz 2>/dev/null | head -1")
BASE_REL="${BASE_L1#${REMOTE_ROOT}/}"
D16_REL="${D16_L1#${REMOTE_ROOT}/}"
L2_REL="$(dirname "${BASE_REL}")/frozen_l2_controller.npz"

if [[ -n "${BASE_L1}" && -n "${D16_L1}" ]]; then
  rivanna_ssh "cd ${REMOTE_ROOT} && module purge && module load miniforge/24.3.0-py3.11 && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate hbac && python -m hbac.scripts.eval_l1_oracle_compare \
    --baseline-l1 ${BASE_REL} \
    --candidate-l1 ${D16_REL} \
    --l2-checkpoint ${L2_REL} \
    --output results/d16_oracle_compare.json"
  echo "D16 oracle compare written on Rivanna"
else
  echo "WARN: missing L1 checkpoints for oracle compare"
fi

if rivanna_ssh "test -f ${REMOTE_ROOT}/results/compose_live_bf040_floor400_scarcity_refined_dpo_v2.json"; then
  echo "D12 refined result exists; skip"
else
  D12R=$(rivanna_ssh "cd ${REMOTE_ROOT} && SHIFT_FRACTION=0.08 SWE_MIN_RESERVE=0.5 sbatch slurm/eval_live_scarcity_boost_refined.sh" | awk '{print $NF}')
  echo "Submitted D12 refined live: ${D12R}"
fi
