#!/bin/bash
#SBATCH -J hbac_vb_par
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Parallel Variant B Stage 3: one budget-curriculum track per array task.
# CPU partition — L1/L2 prototype GRPO is numpy; saves GPU SUs.

set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
pip install -q python-dotenv -e ".[dev]" 2>/dev/null || pip install -q python-dotenv

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"
cd "${HBAC_ROOT}"

bash scripts/rivanna/link_latest_checkpoint.sh

BUDGETS=(0.90 0.75 0.60)
SEEDS=(42 43 44)
BF="${BUDGETS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_TAG="bf${BF//./}_seed${SEED}"

echo "=== Variant B parallel array task ${SLURM_ARRAY_TASK_ID} ==="
echo "budget_fraction=${BF} seed=${SEED} host=$(hostname) date=$(date)"

python -m hbac.scripts.train_variant_b \
  --oracle-path data/oracles \
  --checkpoint checkpoints/variant_a/latest \
  --stage 3 \
  --freeze-l2 \
  --budget-fraction "${BF}" \
  --grpo-groups 16 \
  --num-batches 50 \
  --epochs 12 \
  --use-counterfactual \
  --seed "${SEED}" \
  --output "checkpoints/variant_b/parallel/${RUN_TAG}"

OUT_BASE="checkpoints/variant_b/parallel/${RUN_TAG}/stage3"
RUN_DIR=$(ls -td "${OUT_BASE}"/*/ 2>/dev/null | head -1)
if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: no run dir under ${OUT_BASE}"
  exit 1
fi

python -m hbac.scripts.eval_batch \
  --batches-path "${RUN_DIR}/batches.jsonl" \
  --l1-checkpoint "${RUN_DIR}/level1_policy.npz" \
  --l2-checkpoint checkpoints/variant_a/latest \
  --oracle-path data/oracles \
  --output "results/variant_b_parallel_${RUN_TAG}.json"

echo "Array task ${SLURM_ARRAY_TASK_ID} complete $(date)"
