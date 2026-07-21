#!/bin/bash
# Capability pilot: Qwen3-Coder-Next 4-bit on 1× A6000 (plan-recommended local model).
#SBATCH -J hbac_cap_4bit
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh
pip install -q "bitsandbytes>=0.43.0" 2>/dev/null || true

MODEL="${CAPABILITY_MODEL:-Qwen/Qwen3-Coder-Next}"
export HBAC_LLM_MODEL="${MODEL}"
export HBAC_LOAD_IN_4BIT=1

mkdir -p results/rivanna/capability_pilot_4bit_shards

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
  --checkpoint-dir results/rivanna/capability_pilot_4bit_shards \
  --output results/rivanna/capability_pilot_4bit_uniform.json

python -m hbac.scripts.analyze_capability_pilot \
  --source results/rivanna/capability_pilot_4bit_shards/uniform.json \
  --output results/capability_pilot_4bit_analysis.json

echo "Capability 4bit pilot done $(date)"
