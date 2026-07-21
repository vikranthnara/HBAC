#!/bin/bash
#SBATCH -J hbac_live_d12d16
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

LORA=$(ls -td checkpoints/llm_dpo/*_capability_v2/ | head -1)/model
L1_DIR=$(ls -td checkpoints/phase3_parse_penalty_0.3/*/stage3/ 2>/dev/null | head -1)
if [[ -z "${L1_DIR}" ]]; then
  echo "ERROR: no D16 parse-penalty L1 checkpoint under checkpoints/phase3_parse_penalty_0.3"
  exit 1
fi
L1="${HBAC_L1:-${L1_DIR}/level1_policy.npz}"
L2="${HBAC_L2:-checkpoints/variant_a/latest}"
BATCHES="${HBAC_BATCHES}"

python -m hbac.scripts.eval_compose_live \
  --batches-path "${BATCHES}" \
  --l2-checkpoint "${L2}" \
  --l1-checkpoint "${L1}" \
  --llm-spec "transformers:${HBAC_LLM_MODEL}" \
  --lora-path "${LORA}" \
  --scarcity-boost \
  --budget-fraction 0.40 \
  --live-min-per-task 400 \
  --benchmarks "tau_bench,toolbench,mock,swe_bench" \
  --max-batches 50 \
  --output results/compose_live_bf040_floor400_scarcity_parse_penalty_dpo_v2.json

echo "D12+D16 combined live eval done $(date)"
