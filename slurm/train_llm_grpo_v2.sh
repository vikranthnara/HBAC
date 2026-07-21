#!/bin/bash
#SBATCH -J hbac_grpo_v2
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-1%1
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=END,FAIL

# Array 0: SFT-only baseline (strong format prior)
# Array 1: SFT warmstart + tool-aware GRPO (recommended)
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh

export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export PYTHONUNBUFFERED=1

MODES=(sft_only sft_then_grpo)
TAGS=(sft_only sft_grpo)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

python -m hbac.scripts.train_llm_grpo_v2 \
  --oracle-path data/oracles \
  --model "${HBAC_LLM_MODEL}" \
  --lora-rank 16 \
  --grpo-groups 8 \
  --grpo-epochs 2 \
  --sft-epochs 3 \
  --max-samples 400 \
  --training-mode "${MODE}" \
  --reward-mode tool_aware \
  --max-completion-length 384 \
  --run-suffix "${TAG}" \
  --output checkpoints/llm_grpo_v2

RUN_DIR=$(ls -td checkpoints/llm_grpo_v2/*_"${TAG}"/ | head -1)
python -m hbac.scripts.eval_grpo_format \
  --oracle-path data/oracles \
  --model "${HBAC_LLM_MODEL}" \
  --lora-path "${RUN_DIR}/model" \
  --limit 100 \
  --output "results/grpo_format_${TAG}.json"

echo "GRPO v2 ${TAG} done $(date) -> ${RUN_DIR}"
