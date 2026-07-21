#!/bin/bash
#SBATCH -J hbac_l1_fair
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
bash scripts/rivanna/link_latest_checkpoint.sh

STARVATION_PENALTY="${STARVATION_PENALTY:-0.5}"
HARD_MIN_FRAC="${HARD_MIN_FRAC:-0.15}"
OUT="checkpoints/phase3_fairness_${STARVATION_PENALTY}"

python -m hbac.scripts.run_phase3 \
  --oracle-path data/oracles/real_eval/latest \
  --checkpoint checkpoints/variant_a/latest \
  --output "${OUT}" \
  --grpo-groups 16 \
  --num-batches 50 \
  --epochs 8 \
  --budget-fraction 0.40 \
  --starvation-penalty "${STARVATION_PENALTY}" \
  --hard-min-frac "${HARD_MIN_FRAC}" \
  --skip-stage4 \
  --skip-variant-a

echo "D18 L1 fairness retrain done $(date) -> ${OUT}"
