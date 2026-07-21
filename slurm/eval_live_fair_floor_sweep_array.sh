#!/bin/bash
# Dose-response: hbac_fair vs type_prior across per-task floors (n~300 pilot batches).
#SBATCH -J hbac_fair_floor
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

export PYTHONUNBUFFERED=1

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh
source slurm/_resolve_live_ckpt.sh

export HBAC_LLM_PROVIDER=transformers
export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

FLOORS=(300 350 400 450 500 600)
ALLOCATORS=(hbac_fair type_prior)
FLOOR_IDX=$((SLURM_ARRAY_TASK_ID / 2))
ALLOC_IDX=$((SLURM_ARRAY_TASK_ID % 2))
FLOOR="${FLOORS[$FLOOR_IDX]}"
ALLOC="${ALLOCATORS[$ALLOC_IDX]}"
SHARD_DIR="results/fair_floor_sweep_shards/floor${FLOOR}"

BATCHES="${HBAC_EVAL_BATCHES:-checkpoints/eval_real/batches.jsonl}"
LORA=$(ls -td checkpoints/llm_dpo/*_capability_v2/ 2>/dev/null | head -1)/model

echo "Array ${SLURM_ARRAY_TASK_ID}: floor=${FLOOR} allocator=${ALLOC} $(date)"

python -m hbac.scripts.eval_compose_live \
  --batches-path "${BATCHES}" \
  --l2-checkpoint "${HBAC_L2}" \
  --l1-checkpoint "${HBAC_L1}" \
  --llm-spec "transformers:${HBAC_LLM_MODEL}" \
  --lora-path "${LORA}" \
  --budget-fraction 0.40 \
  --live-min-per-task "${FLOOR}" \
  --benchmarks "tau_bench,toolbench,mock,swe_bench,livecodebench" \
  --max-batches 50 \
  --fairness-reserve \
  --hard-min-frac 0.15 \
  --only-allocator "${ALLOC}" \
  --checkpoint-dir "${SHARD_DIR}" \
  --output "results/fair_floor_sweep.json"

echo "Shard floor=${FLOOR} ${ALLOC} done $(date)"
