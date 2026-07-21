#!/bin/bash
# Option 1: full-GPU device_map=auto (no max_memory cap).
#SBATCH -J hbac_cap_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:2
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh
pip install -q "bitsandbytes>=0.43.0" "accelerate>=0.30.0" 2>/dev/null || true

MODEL="${CAPABILITY_MODEL:-Qwen/Qwen3-Coder-Next}"
SHARD_DIR="${CAPABILITY_SHARD_DIR:-results/rivanna/capability_pilot_2xa100_gpuonly_shards}"
OUT_JSON="${CAPABILITY_OUT_JSON:-results/rivanna/capability_pilot_2xa100_gpuonly_uniform.json}"
ANALYSIS_JSON="${CAPABILITY_ANALYSIS_JSON:-results/capability_pilot_2xa100_gpuonly_analysis.json}"

export HBAC_LLM_MODEL="${MODEL}"
export HBAC_LOAD_IN_4BIT=1
export HBAC_BNB_GPU_ONLY=1
unset HBAC_BNB_CPU_OFFLOAD HBAC_MAX_MEMORY_PER_GPU HBAC_MAX_MEMORY_CPU || true

echo "MODE=gpu_only CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("n_gpu", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu{i}", p.name, f"mem_gb={p.total_memory/1024**3:.1f}")
PY

mkdir -p "${SHARD_DIR}"

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
  --llm-spec "transformers:${MODEL}" \
  --checkpoint-dir "${SHARD_DIR}" \
  --output "${OUT_JSON}"

python -m hbac.scripts.analyze_capability_pilot \
  --source "${SHARD_DIR}/uniform.json" \
  --output "${ANALYSIS_JSON}"

echo "Capability gpu_only pilot done $(date)"
