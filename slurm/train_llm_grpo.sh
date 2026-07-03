#!/bin/bash
#SBATCH -J hbac_llm_grpo
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
module purge
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"
cd "${HBAC_ROOT}"

# shellcheck source=/dev/null
source slurm/_gpu_setup.sh

MODEL="${HBAC_LLM_MODEL}"
python -m hbac.scripts.train_llm_grpo \
  --oracle-path data/oracles \
  --model "${MODEL}" \
  --lora-rank 16 \
  --grpo-groups 8 \
  --num-batches 10 \
  --epochs 2 \
  --output checkpoints/llm_grpo

echo "LLM GRPO done $(date)"
