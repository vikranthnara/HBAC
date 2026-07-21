#!/bin/bash
#SBATCH -J hbac_l1_parse
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

PARSE_PENALTY="${PARSE_PENALTY:-0.3}"
OUT="checkpoints/phase3_parse_penalty_${PARSE_PENALTY}"

python -m hbac.scripts.run_phase3 \
  --oracle-path data/oracles \
  --checkpoint checkpoints/variant_a/latest \
  --output "${OUT}" \
  --grpo-groups 16 \
  --num-batches 30 \
  --epochs 8 \
  --parse-penalty "${PARSE_PENALTY}" \
  --skip-stage4 \
  --skip-variant-a

echo "D16 L1 parse-penalty retrain done $(date) -> ${OUT}"
