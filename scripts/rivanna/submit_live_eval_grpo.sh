#!/usr/bin/env bash
# Live compose eval with TRL GRPO LoRA adapter on frozen base model.
set -euo pipefail

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"

module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export HBAC_LIVE_TAG="${HBAC_LIVE_TAG:-bf040_seed47}"
export HBAC_LORA_PATH="${HBAC_LORA_PATH:-checkpoints/llm_grpo/20260703T080820Z/model}"
export HBAC_LIVE_SUFFIX="${HBAC_LIVE_SUFFIX:-grpo_lora}"

LIVE_ID=$(sbatch --parsable \
  --export=ALL,HBAC_ROOT="${HBAC_ROOT}",HBAC_LLM_MODEL="${HBAC_LLM_MODEL}",HBAC_LIVE_TAG="${HBAC_LIVE_TAG}",HBAC_LORA_PATH="${HBAC_LORA_PATH}",HBAC_LIVE_SUFFIX="${HBAC_LIVE_SUFFIX}" \
  slurm/eval_compose_live_grpo.sh)
echo "live_eval_grpo=${LIVE_ID}"
