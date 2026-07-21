#!/bin/bash
# V3 pilot: live heuristic @ floor=400, n~300 (50 batches) — faster GPU backfill.
#SBATCH -J hbac_v3_pilot
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh
source slurm/_resolve_live_ckpt.sh

export HBAC_LLM_PROVIDER=transformers
export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

BATCHES="${HBAC_EVAL_BATCHES:-checkpoints/eval_real/batches.jsonl}"
LORA=$(ls -td checkpoints/llm_dpo/*_capability_v2/ 2>/dev/null | head -1)/model

python -m hbac.scripts.eval_compose_live \
  --batches-path "${BATCHES}" \
  --l2-checkpoint "${HBAC_L2}" \
  --l1-checkpoint "${HBAC_L1}" \
  --llm-spec "transformers:${HBAC_LLM_MODEL}" \
  --lora-path "${LORA}" \
  --budget-fraction 0.40 \
  --live-min-per-task 400 \
  --benchmarks "tau_bench,toolbench,mock,swe_bench,livecodebench" \
  --max-batches 50 \
  --fairness-reserve \
  --hard-min-frac 0.15 \
  --output "results/compose_live_v3_pilot_floor400_dpo_v2.json"

echo "V3 pilot live done $(date)"
