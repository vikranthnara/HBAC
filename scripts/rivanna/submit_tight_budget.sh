#!/usr/bin/env bash
# Submit 40–50% budget Variant B parallel array (+ optional post-hoc live LLM eval).
# Usage: HBAC_ROOT=/path/to/run bash scripts/rivanna/submit_tight_budget.sh
set -euo pipefail

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"

mkdir -p logs checkpoints/variant_b/parallel_tight results

echo "=== HBAC tight-budget Variant B (40–50%) ==="
echo "Root: ${HBAC_ROOT}"
echo "Host: $(hostname)"
echo "Date: $(date)"

module purge
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
python -c "import hbac; print('preflight OK')"

TIGHT_ID=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}" slurm/variant_b_parallel_tight.sh)
echo "  tight array job: ${TIGHT_ID}"

LIVE_ID=""
if [[ "${SUBMIT_LIVE_EVAL:-1}" == "1" ]]; then
  LIVE_ID=$(sbatch --parsable --dependency=afterok:"${TIGHT_ID}" \
    --export=ALL,HBAC_ROOT="${HBAC_ROOT}" slurm/eval_compose_live.sh)
  echo "  live LLM compose eval (after tight): ${LIVE_ID}"
fi

{
  echo "tight_array=${TIGHT_ID}"
  echo "live_eval=${LIVE_ID}"
  echo "submitted=$(date -u +%Y%m%dT%H%M%SZ)"
} > logs/tight_budget_jobids.txt

cat logs/tight_budget_jobids.txt
