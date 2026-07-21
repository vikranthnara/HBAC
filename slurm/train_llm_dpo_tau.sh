#!/bin/bash
#SBATCH -J hbac_dpo_tau
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh

export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

python -m hbac.scripts.collect_stub_oracles --output data/oracles/stub_live

python -m hbac.scripts.train_llm_dpo \
  --oracle-path data/oracles \
  --benchmark tau_bench \
  --model "${HBAC_LLM_MODEL}" \
  --max-pairs 300 \
  --epochs 3 \
  --sft-epochs 3 \
  --reject-modes wrong_tool \
  --output checkpoints/llm_dpo \
  --run-suffix tau_v3

RUN_DIR=$(ls -td checkpoints/llm_dpo/*_tau_v3/ | head -1)
python -m hbac.scripts.eval_grpo_format \
  --oracle-path data/oracles \
  --model "${HBAC_LLM_MODEL}" \
  --lora-path "${RUN_DIR}/model" \
  --limit 100 \
  --output results/grpo_format_dpo_tau_v3.json

echo "DPO tau v3 done $(date) -> ${RUN_DIR}"
