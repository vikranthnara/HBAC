#!/usr/bin/env bash
# Resubmit GPU jobs: live compose eval + Phase 3b LLM GRPO (no re-train).
# Usage: HBAC_ROOT=/path/to/run bash scripts/rivanna/submit_gpu_jobs.sh
set -euo pipefail

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"

module purge
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

echo "=== HBAC GPU job resubmit ==="
echo "Root: ${HBAC_ROOT}"
echo "Model: ${HBAC_LLM_MODEL}"

LIVE_ID=$(sbatch --parsable \
  --export=ALL,HBAC_ROOT="${HBAC_ROOT}",HBAC_LLM_MODEL="${HBAC_LLM_MODEL}" \
  slurm/eval_compose_live.sh)
echo "  live compose eval: ${LIVE_ID}"

GRPO_ID=$(sbatch --parsable \
  --export=ALL,HBAC_ROOT="${HBAC_ROOT}",HBAC_LLM_MODEL="${HBAC_LLM_MODEL}" \
  slurm/train_llm_grpo.sh)
echo "  llm grpo: ${GRPO_ID}"

{
  echo "live_eval=${LIVE_ID}"
  echo "llm_grpo=${GRPO_ID}"
  echo "model=${HBAC_LLM_MODEL}"
  echo "submitted=$(date -u +%Y%m%dT%H%M%SZ)"
} > logs/gpu_jobids.txt

cat logs/gpu_jobids.txt
