# Shared GPU job setup: conda, vLLM, ungated default model.
# Source from slurm/eval_compose_live.sh and slurm/train_llm_grpo.sh.

HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export HBAC_LLM_PROVIDER=vllm
export HBAC_LLM_BASE_URL="${HBAC_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export HBAC_LLM_MODEL
export HF_HOME="${HF_HOME:-${HBAC_ROOT:-/tmp}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
mkdir -p "${HF_HOME}"

pip install -q python-dotenv -e ".[gpu]"
pip install -q "vllm>=0.6.0"

python -c "import vllm; print('vllm', vllm.__version__)"
nvidia-smi -L 2>/dev/null || echo "WARN: nvidia-smi unavailable"

start_vllm() {
  if curl -sf "${HBAC_LLM_BASE_URL}/models" >/dev/null 2>&1; then
    echo "vLLM already responding at ${HBAC_LLM_BASE_URL}"
    return 0
  fi
  echo "Starting vLLM with ${HBAC_LLM_MODEL}..."
  python -m vllm.entrypoints.openai.api_server \
    --model "${HBAC_LLM_MODEL}" \
    --dtype float16 \
    --max-model-len 4096 \
    --port 8000 \
    --host 127.0.0.1 &
  VLLM_PID=$!
  for i in $(seq 1 90); do
    if curl -sf "${HBAC_LLM_BASE_URL}/models" >/dev/null 2>&1; then
      echo "vLLM ready after ${i}0s"
      return 0
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "ERROR: vLLM process exited before ready"
      return 1
    fi
    sleep 10
  done
  echo "ERROR: vLLM did not become ready in 15 minutes"
  return 1
}
