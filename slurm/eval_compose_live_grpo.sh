#!/bin/bash
#SBATCH -J hbac_live_grpo
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh

export HBAC_LLM_PROVIDER=transformers
export HBAC_LLM_MODEL="${HBAC_LLM_MODEL}"
export PYTHONUNBUFFERED=1
HBAC_LIVE_TAG="${HBAC_LIVE_TAG:-bf040_seed47}"
HBAC_LORA_PATH="${HBAC_LORA_PATH:-checkpoints/llm_grpo/20260703T080820Z/model}"
HBAC_LIVE_SUFFIX="${HBAC_LIVE_SUFFIX:-grpo_lora}"
HBAC_MAX_BATCHES="${HBAC_MAX_BATCHES:-50}"
HBAC_LIVE_MIN_PER_TASK="${HBAC_LIVE_MIN_PER_TASK:-600}"

RUN_DIR=$(ls -td "checkpoints/variant_b/parallel_tight/${HBAC_LIVE_TAG}/stage3"/*/ 2>/dev/null | head -1)
if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: checkpoint missing for ${HBAC_LIVE_TAG}"
  exit 1
fi

echo "Live GRPO eval tag=${HBAC_LIVE_TAG} lora=${HBAC_LORA_PATH}"

python -m hbac.scripts.eval_compose_live \
  --batches-path "${RUN_DIR}/batches.jsonl" \
  --l2-checkpoint "${RUN_DIR}/frozen_l2_controller.npz" \
  --l1-checkpoint "${RUN_DIR}/level1_policy.npz" \
  --llm-spec "transformers:${HBAC_LLM_MODEL}" \
  --lora-path "${HBAC_LORA_PATH}" \
  --benchmarks "tau_bench,toolbench,mock,swe_bench" \
  --max-batches "${HBAC_MAX_BATCHES}" \
  --live-min-per-task "${HBAC_LIVE_MIN_PER_TASK}" \
  --output "results/compose_live_${HBAC_LIVE_TAG}_${HBAC_LIVE_SUFFIX}.json"

echo "Live GRPO eval done $(date)"
