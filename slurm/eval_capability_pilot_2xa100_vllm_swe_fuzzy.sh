#!/bin/bash
# One-shot SWE salvage: fuzzy gold-line grading + aligned ReAct prompt.
#SBATCH -J hbac_cap_swefuzzy
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
SHARD_DIR=results/rivanna/capability_pilot_vllm_swe_fuzzy_shards
OUT_JSON=results/rivanna/capability_pilot_vllm_swe_fuzzy_uniform.json
ANALYSIS_JSON=results/capability_pilot_vllm_swe_fuzzy_analysis.json
VLLM_LOG=logs/vllm_capability_swe_fuzzy_${SLURM_JOB_ID:-local}.log

export HF_HOME="${HF_HOME:-${HBAC_ROOT}/.cache/huggingface}"
export HBAC_SWE_LOCAL_GRADE=fuzzy
mkdir -p "${HF_HOME}" logs "${SHARD_DIR}"

echo "MODE=vllm_swe_fuzzy MODEL=${MODEL} GRADE=${HBAC_SWE_LOCAL_GRADE}"
python -c "from hbac.baselines.react import ReActRunner; assert 'str_replace_editor' in ReActRunner.system_prompt_for_benchmark('swe_bench')"

pip install -q 'vllm>=0.15.0'
python -c "import vllm; print('vllm', vllm.__version__)"

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

vllm serve "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

READY=0
for i in $(seq 1 180); do
  kill -0 "${VLLM_PID}" 2>/dev/null || { tail -80 "${VLLM_LOG}"; exit 1; }
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; echo "vLLM ready ~$((i*10))s"; break
  fi
  sleep 10
done
[[ "${READY}" -eq 1 ]] || { tail -80 "${VLLM_LOG}"; exit 1; }

export HBAC_LLM_PROVIDER=vllm HBAC_LLM_MODEL="${MODEL}"
export HBAC_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export HBAC_LLM_API_KEY=EMPTY OPENAI_API_KEY=EMPTY

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

echo "SWE fuzzy salvage done $(date)"
