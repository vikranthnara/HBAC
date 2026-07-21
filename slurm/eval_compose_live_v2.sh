#!/bin/bash
#SBATCH -J hbac_live_v2
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%1
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh

export HBAC_LLM_PROVIDER=transformers
export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export PYTHONUNBUFFERED=1

TAGS=(sft_only sft_grpo)
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR=$(ls -td checkpoints/llm_grpo_v2/*_"${TAG}"/ | head -1)
HBAC_LIVE_TAG="${HBAC_LIVE_TAG:-bf040_seed47}"

CKPT_DIR=$(ls -td "checkpoints/variant_b/parallel_tight/${HBAC_LIVE_TAG}/stage3"/*/ 2>/dev/null | head -1)
if [[ -z "${CKPT_DIR}" || -z "${RUN_DIR}" ]]; then
  echo "ERROR: missing checkpoint tag=${TAG} ckpt=${CKPT_DIR} lora=${RUN_DIR}"
  exit 1
fi

python -m hbac.scripts.eval_compose_live \
  --batches-path "${CKPT_DIR}/batches.jsonl" \
  --l2-checkpoint "${CKPT_DIR}/frozen_l2_controller.npz" \
  --l1-checkpoint "${CKPT_DIR}/level1_policy.npz" \
  --llm-spec "transformers:${HBAC_LLM_MODEL}" \
  --lora-path "${RUN_DIR}/model" \
  --benchmarks "tau_bench,toolbench,mock,swe_bench" \
  --max-batches 50 \
  --live-min-per-task 600 \
  --output "results/compose_live_${HBAC_LIVE_TAG}_v2_${TAG}.json"

echo "Live v2 ${TAG} done $(date)"
