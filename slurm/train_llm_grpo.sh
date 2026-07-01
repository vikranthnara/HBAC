#!/bin/bash
#SBATCH -J hbac_llm_grpo
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=a100_40gb
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
module purge
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

cd "${HBAC_ROOT:-/standard/liverobotics/hbac}"

export HBAC_LLM_PROVIDER=vllm
export HBAC_LLM_BASE_URL="${HBAC_LLM_BASE_URL:-http://localhost:8000/v1}"
export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

pip install -q -e ".[gpu]"

# Start vLLM if not already running (single-node)
if ! curl -sf "${HBAC_LLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "Starting vLLM server..."
  python -m vllm.entrypoints.openai.api_server \
    --model "${HBAC_LLM_MODEL}" \
    --dtype float16 \
    --max-model-len 4096 \
    --port 8000 &
  sleep 120
fi

python -m hbac.scripts.train_llm_grpo \
  --oracle-path data/oracles \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --lora-rank 16 \
  --grpo-groups 8 \
  --num-batches 10 \
  --epochs 2 \
  --output checkpoints/llm_grpo

echo "LLM GRPO done $(date)"
