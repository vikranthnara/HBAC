#!/bin/bash
#SBATCH -J hbac_vb_tight
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Tight-budget Variant B Stage 3: 50% / 45% / 40% oracle-token fractions.

set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
pip install -q python-dotenv -e ".[dev]" 2>/dev/null || pip install -q python-dotenv

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"
cd "${HBAC_ROOT}"

BUDGETS=(0.50 0.45 0.40)
SEEDS=(45 46 47)
BF="${BUDGETS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_TAG="bf${BF//./}_seed${SEED}"

echo "=== Variant B TIGHT array task ${SLURM_ARRAY_TASK_ID} ==="
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
  --output "checkpoints/variant_b/parallel_tight/${RUN_TAG}"

OUT_BASE="checkpoints/variant_b/parallel_tight/${RUN_TAG}/stage3"
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
  --output "results/variant_b_tight_${RUN_TAG}.json"

python -m hbac.scripts.eval_compose \
  --batches-path "${RUN_DIR}/batches.jsonl" \
  --l2-checkpoint "${RUN_DIR}/frozen_l2_controller.npz" \
  --l1-checkpoint "${RUN_DIR}/level1_policy.npz" \
  --oracle-path data/oracles \
  --output "results/compose_tight_${RUN_TAG}.json"

echo "Tight array task ${SLURM_ARRAY_TASK_ID} complete $(date)"
