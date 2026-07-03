#!/bin/bash
#SBATCH -J hbac_live_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Live-LLM compose eval on best tight-budget checkpoint (vLLM on-node).

set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"
cd "${HBAC_ROOT}"

# shellcheck source=/dev/null
source slurm/_gpu_setup.sh
start_vllm

BEST_TAG=""
BEST_REWARD=-1
for f in results/compose_tight_bf050_seed45.json \
         results/compose_tight_bf045_seed46.json \
         results/compose_tight_bf040_seed47.json; do
  if [[ -f "${f}" ]]; then
    R=$(python -c "import json; d=json.load(open('${f}')); print(d['hbac_joint']['mean_batch_reward'])")
    if python -c "import sys; sys.exit(0 if float('${R}') > ${BEST_REWARD} else 1)"; then
      BEST_REWARD="${R}"
      BEST_TAG=$(basename "${f}" | sed 's/compose_tight_//;s/.json//')
    fi
  fi
done

if [[ -z "${BEST_TAG}" ]]; then
  BEST_TAG="bf040_seed47"
  echo "WARN: no compose_tight results; defaulting to ${BEST_TAG}"
fi

RUN_DIR=$(ls -td "checkpoints/variant_b/parallel_tight/${BEST_TAG}/stage3"/*/ 2>/dev/null | head -1)
if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: checkpoint missing for ${BEST_TAG}"
  exit 1
fi

echo "Live eval on ${BEST_TAG} run=${RUN_DIR} model=${HBAC_LLM_MODEL}"

python -m hbac.scripts.eval_compose_live \
  --batches-path "${RUN_DIR}/batches.jsonl" \
  --l2-checkpoint "${RUN_DIR}/frozen_l2_controller.npz" \
  --l1-checkpoint "${RUN_DIR}/level1_policy.npz" \
  --llm-spec "vllm:${HBAC_LLM_MODEL}" \
  --benchmarks "tau_bench,toolbench,mock" \
  --max-batches 15 \
  --output "results/compose_live_${BEST_TAG}.json"

echo "Live compose eval done $(date)"
