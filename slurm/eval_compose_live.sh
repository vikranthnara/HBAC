#!/bin/bash
#SBATCH -J hbac_live_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=a100_40gb
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Live-LLM compose eval on best tight-budget checkpoint (vLLM on-node).

set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
pip install -q python-dotenv -e ".[gpu]" 2>/dev/null || pip install -q -e ".[gpu]"

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"
cd "${HBAC_ROOT}"

export HBAC_LLM_PROVIDER=vllm
export HBAC_LLM_BASE_URL="${HBAC_LLM_BASE_URL:-http://localhost:8000/v1}"
export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

if ! curl -sf "${HBAC_LLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "Starting vLLM..."
  python -m vllm.entrypoints.openai.api_server \
    --model "${HBAC_LLM_MODEL}" \
    --dtype float16 \
    --max-model-len 4096 \
    --port 8000 &
  for i in $(seq 1 60); do
    curl -sf "${HBAC_LLM_BASE_URL}/models" >/dev/null 2>&1 && break
    sleep 10
  done
fi

# Pick tight track with highest mean_batch_reward from oracle compose results
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
  echo "WARN: no compose_tight results yet; defaulting to ${BEST_TAG}"
fi

RUN_DIR=$(ls -td "checkpoints/variant_b/parallel_tight/${BEST_TAG}/stage3"/*/ 2>/dev/null | head -1)
if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: checkpoint missing for ${BEST_TAG}"
  exit 1
fi

echo "Live eval on ${BEST_TAG} run=${RUN_DIR}"

python -m hbac.scripts.eval_compose_live \
  --batches-path "${RUN_DIR}/batches.jsonl" \
  --l2-checkpoint "${RUN_DIR}/frozen_l2_controller.npz" \
  --l1-checkpoint "${RUN_DIR}/level1_policy.npz" \
  --llm-spec "vllm:${HBAC_LLM_MODEL}" \
  --benchmarks "tau_bench,toolbench,mock" \
  --max-batches 15 \
  --output "results/compose_live_${BEST_TAG}.json"

echo "Live compose eval done $(date)"
