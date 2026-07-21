#!/usr/bin/env bash
# Run on Rivanna login node after SSH. Submits all pending impact-loop blockers.
#   ssh eyu8ps@login.hpc.virginia.edu
#   cd /standard/liverobotics/hbac-run-20260630T183941Z
#   bash scripts/rivanna/execute_blockers.sh
set -euo pipefail

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"

module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
mkdir -p logs results

echo "=== HBAC execute blockers @ $(date -u) ==="
echo "Root: ${HBAC_ROOT}"

# Ensure variant_a symlink
if [[ -x scripts/rivanna/link_latest_checkpoint.sh ]]; then
  bash scripts/rivanna/link_latest_checkpoint.sh checkpoints/variant_a || true
fi

# 1. Live retrain eval (bf040 tight checkpoint)
if [[ ! -f results/compose_live_bf040_seed47_retrain.json ]]; then
  echo "[1/3] Submitting live retrain eval..."
  export HBAC_LIVE_TAG=bf040_seed47 HBAC_LIVE_SUFFIX=retrain
  LIVE_ID=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}",HBAC_LIVE_TAG,HBAC_LIVE_SUFFIX,HBAC_LLM_MODEL \
    slurm/eval_compose_live.sh)
  echo "  live_retrain=${LIVE_ID}"
else
  echo "[1/3] SKIP live retrain — results/compose_live_bf040_seed47_retrain.json exists"
fi

# 2. H6 long-scale (150 batches)
if [[ ! -f results/h6_long_summary.json ]]; then
  echo "[2/3] Submitting H6 long ablation..."
  H6_ID=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}" slurm/variant_b_h6_long.sh)
  echo "  h6_long=${H6_ID}"
else
  echo "[2/3] SKIP H6 long — results/h6_long_summary.json exists"
fi

# 3. GRPO LoRA live eval
if [[ ! -f results/compose_live_bf040_seed47_grpo_lora.json ]]; then
  echo "[3/3] Submitting GRPO LoRA live eval..."
  export HBAC_LORA_PATH="${HBAC_LORA_PATH:-checkpoints/llm_grpo/20260703T080820Z/model}"
  export HBAC_LIVE_SUFFIX=grpo_lora HBAC_LIVE_TAG=bf040_seed47
  GRPO_ID=$(sbatch --parsable \
    --export=ALL,HBAC_ROOT="${HBAC_ROOT}",HBAC_LORA_PATH,HBAC_LIVE_SUFFIX,HBAC_LIVE_TAG,HBAC_LLM_MODEL \
    slurm/eval_compose_live_grpo.sh)
  echo "  live_grpo_lora=${GRPO_ID}"
else
  echo "[3/3] SKIP GRPO LoRA — results/compose_live_bf040_seed47_grpo_lora.json exists"
fi

{
  echo "executed=$(date -u +%Y%m%dT%H%M%SZ)"
  squeue -u "$(whoami)" 2>/dev/null || true
} | tee logs/blockers_executed.txt

echo ""
echo "Monitor: squeue -u \$USER"
echo "When done, on laptop: bash scripts/rivanna/pull_from_rivanna.sh && bash scripts/run_impact_loop.sh"
