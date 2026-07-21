#!/usr/bin/env bash
# Submit all high-impact Rivanna jobs from the impact loop (pending steps).
set -euo pipefail

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
export HBAC_ROOT

cd "${HBAC_ROOT}"
mkdir -p logs
: > logs/impact_wave_jobids.txt

echo "=== Impact wave submit @ ${HBAC_ROOT} ==="

# 1. Live eval on retrained bf040 (if not already done)
if [[ ! -f results/compose_live_bf040_seed47_retrain.json ]]; then
  export HBAC_LIVE_TAG=bf040_seed47
  export HBAC_LIVE_SUFFIX=retrain
  LIVE_RETRAIN=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}",HBAC_LIVE_TAG,HBAC_LIVE_SUFFIX slurm/eval_compose_live.sh)
  echo "live_retrain=${LIVE_RETRAIN}" | tee -a logs/impact_wave_jobids.txt
else
  echo "live_retrain=skip (results exist)"
fi

# 2. H6 long-scale counterfactual ablation
H6_ID=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}" slurm/variant_b_h6_long.sh)
echo "h6_long=${H6_ID}" | tee -a logs/impact_wave_jobids.txt

# 3. GRPO LoRA live eval
export HBAC_LORA_PATH="${HBAC_LORA_PATH:-checkpoints/llm_grpo/20260703T080820Z/model}"
export HBAC_LIVE_SUFFIX=grpo_lora
GRPO_LIVE=$(sbatch --parsable \
  --export=ALL,HBAC_ROOT="${HBAC_ROOT}",HBAC_LORA_PATH,HBAC_LIVE_SUFFIX \
  slurm/eval_compose_live_grpo.sh)
echo "live_grpo_lora=${GRPO_LIVE}" | tee -a logs/impact_wave_jobids.txt

echo "submitted=$(date -u +%Y%m%dT%H%M%SZ)" >> logs/impact_wave_jobids.txt
cat logs/impact_wave_jobids.txt
