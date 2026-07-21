#!/bin/bash
#SBATCH -J hbac_live_ctrl
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

python -m hbac.scripts.eval_compose_live \
  --batches-path "${HBAC_BATCHES}" \
  --l2-checkpoint "${HBAC_L2}" \
  --l1-checkpoint "${HBAC_L1}" \
  --llm-spec "transformers:${HBAC_LLM_MODEL}" \
  --lora-path "${LORA}" \
  --runner controller \
  --benchmarks "tau_bench,toolbench,mock,swe_bench" \
  --max-batches 50 \
  --live-min-per-task 600 \
  --output results/compose_live_bf040_controller_runner_dpo_v2.json

echo "ControllerRunner live eval done $(date)"
