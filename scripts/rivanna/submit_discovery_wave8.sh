#!/usr/bin/env bash
# Wave 8: Optimization-first runs (D12 confirm, D16 retrain, D14 ROI skip).
# Run AFTER current jobs complete and results pulled.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
bash "${SCRIPT_DIR}/sync_to_rivanna.sh"

# D12 already submitted as 16772086; re-submit only if missing
if rivanna_ssh "test -f ${REMOTE_ROOT}/results/compose_live_bf040_floor400_scarcity_boost_dpo_v2.json"; then
  echo "D12 scarcity boost result exists; skip re-submit"
else
  SCARCITY=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_scarcity_boost.sh" | awk '{print $NF}')
  echo "Submitted D12 scarcity boost: ${SCARCITY}"
fi

# D16: L1 retrain with parse penalty (oracle GRPO, ~8h GPU)
D16_CKPT=$(rivanna_ssh "ls -td ${REMOTE_ROOT}/checkpoints/phase3_parse_penalty_0.3/*/stage3/level1_policy.npz 2>/dev/null | head -1" || true)
if [[ -n "${D16_CKPT}" ]]; then
  echo "D16 L1 parse-penalty checkpoint exists: ${D16_CKPT}"
else
  D16=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/train_l1_parse_penalty.sh" | awk '{print $NF}')
  echo "Submitted D16 L1 parse-penalty retrain: ${D16}"
fi

# D14: ROI skip at floor=300
if rivanna_ssh "test -f ${REMOTE_ROOT}/results/compose_live_bf040_floor300_roi_skip_dpo_v2.json"; then
  echo "D14 ROI skip result exists; skip re-submit"
else
  D14=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_roi_skip.sh" | awk '{print $NF}')
  echo "Submitted D14 ROI skip live (floor=300): ${D14}"
fi

# D12+D16 combined eval — depends on D16 checkpoint
if [[ -n "${D16_CKPT:-}" ]] || rivanna_ssh "ls -td ${REMOTE_ROOT}/checkpoints/phase3_parse_penalty_0.3/*/stage3/level1_policy.npz 2>/dev/null | head -1 | grep -q ."; then
  if rivanna_ssh "test -f ${REMOTE_ROOT}/results/compose_live_bf040_floor400_scarcity_parse_penalty_dpo_v2.json"; then
    echo "D12+D16 combined result exists; skip"
  else
    D3=$(rivanna_ssh "cd ${REMOTE_ROOT} && sbatch slurm/eval_live_scarcity_parse_penalty.sh" | awk '{print $NF}')
    echo "Submitted D12+D16 combined eval: ${D3}"
  fi
else
  echo "D12+D16 combined: waiting on D16 checkpoint"
fi

echo "Wave 8 queued. Monitor: squeue -u \$(whoami)"
