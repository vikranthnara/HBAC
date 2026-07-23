#!/bin/bash
# Small Coder-Next live slice: hbac_d18 vs type_prior on LCB+τ+toolbench (no SWE).
# Must land on A100 80GB (FP8 failed on 40GB — job 17192332).
#SBATCH -J hbac_cn_live80
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:2
#SBATCH -C a100_80gb
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
SHARD_DIR=results/rivanna/coder_next_live_slice_shards
MAX_BATCHES="${HBAC_MAX_BATCHES:-20}"
VLLM_LOG=logs/vllm_coder_next_live_${SLURM_JOB_ID:-local}.log

L1="checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz"
L1_D18="checkpoints/phase3_fairness_0.5/20260706T220026Z/stage3/level1_policy.npz"
L2="checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/frozen_l2_controller.npz"

export HF_HOME="${HF_HOME:-${HBAC_ROOT}/.cache/huggingface}"
mkdir -p "${HF_HOME}" logs "${SHARD_DIR}"

python - <<'PY'
import sys
import torch
n = torch.cuda.device_count()
print("n_gpu", n)
ok = True
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    gb = p.total_memory / 1024**3
    print(f"gpu{i}", p.name, f"mem_gb={gb:.1f}")
    if gb < 70:
        ok = False
if n < 2 or not ok:
    print("ERROR: need 2x A100-80GB (got insufficient VRAM); refusing to start vLLM", file=sys.stderr)
    sys.exit(2)
PY

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

for KEY in hbac_d18 type_prior; do
  echo "=== allocator=${KEY} max_batches=${MAX_BATCHES} ==="
  python -m hbac.scripts.eval_compose_live \
    --batches-path checkpoints/eval_n1000/batches.jsonl \
    --l1-checkpoint "${L1}" \
    --l1-checkpoint-d18 "${L1_D18}" \
    --l2-checkpoint "${L2}" \
    --benchmarks livecodebench,tau_bench,toolbench \
    --max-batches "${MAX_BATCHES}" \
    --live-min-per-task 400 \
    --budget-fraction 0.4 \
    --only-allocator "${KEY}" \
    --save-per-task \
    --llm-spec "vllm:${MODEL}" \
    --checkpoint-dir "${SHARD_DIR}" \
    --output "results/rivanna/coder_next_live_slice_${KEY}.json"
done

python -m hbac.scripts.analyze_paired_allocators \
  --shard-dir "${SHARD_DIR}" \
  --pairs "hbac_d18:type_prior" \
  --output results/paired_allocator_analysis_coder_next_slice.json

echo "Coder-Next live slice (80GB) done $(date)"
