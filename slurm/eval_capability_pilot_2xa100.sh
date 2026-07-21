#!/bin/bash
# Capability pilot: Qwen3-Coder-Next 4-bit sharded across 2× A100.
# Retry: previous run failed because max_memory advertised CPU + 38GiB/GPU cap.
#SBATCH -J hbac_cap_2a100
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
export HBAC_LLM_MODEL="${MODEL}"
export HBAC_LOAD_IN_4BIT=1
# Leave HBAC_MAX_MEMORY_PER_GPU unset → loader uses ~90% of each GPU VRAM.
# Do not set HBAC_BNB_CPU_OFFLOAD unless intentionally allowing CPU spill.
unset HBAC_MAX_MEMORY_PER_GPU HBAC_MAX_MEMORY_CPU HBAC_BNB_CPU_OFFLOAD || true

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("n_gpu", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"gpu{i}", props.name, f"mem_gb={props.total_memory/1024**3:.1f}")
PY

mkdir -p results/rivanna/capability_pilot_2xa100_shards

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
  --checkpoint-dir results/rivanna/capability_pilot_2xa100_shards \
  --output results/rivanna/capability_pilot_2xa100_uniform.json

python -m hbac.scripts.analyze_capability_pilot \
  --source results/rivanna/capability_pilot_2xa100_shards/uniform.json \
  --output results/capability_pilot_2xa100_analysis.json

echo "Capability 2xA100 pilot done $(date)"
