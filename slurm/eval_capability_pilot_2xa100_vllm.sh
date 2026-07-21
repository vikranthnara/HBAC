#!/bin/bash
# Capability pilot via vLLM (Qwen3-Coder-Next-FP8, TP=2 on 2× A100 80GB).
# Transformers+bnb failed; Qwen requires vllm>=0.15 for this model family.
#SBATCH -J hbac_cap_vllm
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:2
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh

MODEL="${CAPABILITY_MODEL:-Qwen/Qwen3-Coder-Next-FP8}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
TP="${VLLM_TP_SIZE:-2}"
SHARD_DIR="${CAPABILITY_SHARD_DIR:-results/rivanna/capability_pilot_2xa100_vllm_shards}"
OUT_JSON="${CAPABILITY_OUT_JSON:-results/rivanna/capability_pilot_2xa100_vllm_uniform.json}"
ANALYSIS_JSON="${CAPABILITY_ANALYSIS_JSON:-results/capability_pilot_2xa100_vllm_analysis.json}"
VLLM_LOG="${CAPABILITY_VLLM_LOG:-logs/vllm_capability_pilot_${SLURM_JOB_ID:-local}.log}"

export HF_HOME="${HF_HOME:-${HBAC_ROOT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
mkdir -p "${HF_HOME}" logs "${SHARD_DIR}"

echo "MODE=vllm MODEL=${MODEL} TP=${TP} MAX_LEN=${MAX_LEN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("n_gpu", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu{i}", p.name, f"mem_gb={p.total_memory/1024**3:.1f}")
PY

echo "Installing vLLM (>=0.15) ..."
pip install -q 'vllm>=0.15.0'
python -c "import vllm; print('vllm', vllm.__version__)"

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting vLLM server -> ${VLLM_LOG}"
# FP8 fits 2×80GB; bf16 ~160GB weights alone leaves no KV headroom.
# Short context is enough for capability pilot (SWE/LCB snippets).
vllm serve "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM on :${PORT} (pid=${VLLM_PID}) ..."
READY=0
for i in $(seq 1 180); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited early; last 80 log lines:"
    tail -80 "${VLLM_LOG}" || true
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    echo "vLLM ready after ~$((i * 10))s"
    break
  fi
  sleep 10
done
if [[ "${READY}" -ne 1 ]]; then
  echo "vLLM failed to become ready; last 80 log lines:"
  tail -80 "${VLLM_LOG}" || true
  exit 1
fi

export HBAC_LLM_PROVIDER=vllm
export HBAC_LLM_MODEL="${MODEL}"
export HBAC_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export HBAC_LLM_API_KEY=EMPTY
export OPENAI_API_KEY=EMPTY

python -m hbac.scripts.eval_compose_live \
  --batches-path checkpoints/eval_n1000/batches.jsonl \
  --l2-checkpoint checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/frozen_l2_controller.npz \
  --l1-checkpoint checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz \
  --benchmarks livecodebench,swe_bench \
  --max-batches 10 \
  --live-min-per-task 400 \
  --budget-fraction 0.4 \
  --only-allocator uniform \
  --save-per-task \
  --llm-spec "vllm:${MODEL}" \
  --checkpoint-dir "${SHARD_DIR}" \
  --output "${OUT_JSON}"

python -m hbac.scripts.analyze_capability_pilot \
  --source "${SHARD_DIR}/uniform.json" \
  --output "${ANALYSIS_JSON}"

echo "Capability vLLM pilot done $(date)"
