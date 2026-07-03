#!/usr/bin/env bash
# Submit post-Phase-3 next steps on Rivanna: tight retrain, live eval, LLM GRPO.
set -euo pipefail

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"

module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export HBAC_LIVE_TAG="${HBAC_LIVE_TAG:-bf040_seed47}"
export HBAC_MAX_BATCHES="${HBAC_MAX_BATCHES:-50}"
export HBAC_LIVE_MIN_PER_TASK="${HBAC_LIVE_MIN_PER_TASK:-600}"

mkdir -p logs

echo "=== HBAC next-steps submit ==="
echo "Root: ${HBAC_ROOT}"
echo "Model: ${HBAC_LLM_MODEL}"
echo "Live tag: ${HBAC_LIVE_TAG}"

TIGHT_ID=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}" slurm/variant_b_parallel_tight.sh)
echo "  tight retrain (40/45/50%): ${TIGHT_ID}"

LIVE_ID=$(sbatch --parsable \
  --export=ALL,HBAC_ROOT="${HBAC_ROOT}",HBAC_LLM_MODEL="${HBAC_LLM_MODEL}",HBAC_LIVE_TAG="${HBAC_LIVE_TAG}",HBAC_MAX_BATCHES="${HBAC_MAX_BATCHES}",HBAC_LIVE_MIN_PER_TASK="${HBAC_LIVE_MIN_PER_TASK}" \
  slurm/eval_compose_live.sh)
echo "  live compose eval: ${LIVE_ID}"

GRPO_ID=$(sbatch --parsable \
  --export=ALL,HBAC_ROOT="${HBAC_ROOT}",HBAC_LLM_MODEL="${HBAC_LLM_MODEL}" \
  slurm/train_llm_grpo.sh)
echo "  llm grpo: ${GRPO_ID}"

{
  echo "tight_retrain=${TIGHT_ID}"
  echo "live_eval=${LIVE_ID}"
  echo "llm_grpo=${GRPO_ID}"
  echo "live_tag=${HBAC_LIVE_TAG}"
  echo "model=${HBAC_LLM_MODEL}"
  echo "submitted=$(date -u +%Y%m%dT%H%M%SZ)"
} > logs/next_steps_jobids.txt

cat logs/next_steps_jobids.txt
