#!/bin/bash
# V3 live matrix with LCB-holdout DPO LoRA (primary allocators).
#SBATCH -J hbac_v3_hold
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh

ALLOCATORS=(hbac_d18 type_prior hbac_joint hbac_guardrail uniform)
KEY="${ALLOCATORS[$SLURM_ARRAY_TASK_ID]}"

L1="checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz"
L1_D18="checkpoints/phase3_fairness_0.5/20260706T220026Z/stage3/level1_policy.npz"
L2="checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/frozen_l2_controller.npz"
LORA="${V3_LORA_PATH:-$(ls -td checkpoints/llm_dpo/*_capability_holdout/ 2>/dev/null | head -1)/model}"

if [[ ! -d "${LORA}" ]]; then
  echo "ERROR: missing holdout LoRA at ${LORA}"
  exit 1
fi

echo "Holdout live: allocator=${KEY} lora=${LORA} model=${HBAC_LLM_MODEL}"

python -m hbac.scripts.eval_compose_live \
  --batches-path checkpoints/eval_n1000/batches.jsonl \
  --l1-checkpoint "${L1}" \
  --l1-checkpoint-d18 "${L1_D18}" \
  --l2-checkpoint "${L2}" \
  --benchmarks livecodebench,swe_bench,tau_bench,toolbench \
  --max-batches 0 \
  --live-min-per-task 400 \
  --budget-fraction 0.4 \
  --lora-path "${LORA}" \
  --llm-spec "transformers:${HBAC_LLM_MODEL}" \
  --only-allocator "${KEY}" \
  --save-per-task \
  --checkpoint-dir results/rivanna/v3_live_holdout_shards

echo "Holdout live shard ${KEY} done $(date)"
