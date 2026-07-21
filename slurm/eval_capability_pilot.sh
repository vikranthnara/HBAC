#!/bin/bash
# Capability pilot: uniform allocation only, SWE+LCB, n~100 tasks.
#SBATCH -J hbac_cap_pilot
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh

MODEL="${CAPABILITY_MODEL:-Qwen/Qwen3-Coder-Next}"
LORA="${CAPABILITY_LORA:-}"
export HBAC_LLM_MODEL="${MODEL}"

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
  ${LORA:+--lora-path "$LORA"} \
  --checkpoint-dir results/rivanna/capability_pilot_shards \
  --output results/rivanna/capability_pilot_uniform.json

python -m hbac.scripts.analyze_capability_pilot \
  --source results/rivanna/capability_pilot_shards/uniform.json

echo "Capability pilot done $(date)"
