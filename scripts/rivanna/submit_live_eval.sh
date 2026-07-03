#!/usr/bin/env bash
# Resubmit live compose eval only (e.g. after V100 failure).
set -euo pipefail
HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LIVE_ID=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}",HBAC_LLM_MODEL="${HBAC_LLM_MODEL}" slurm/eval_compose_live.sh)
echo "live_eval=${LIVE_ID}"
