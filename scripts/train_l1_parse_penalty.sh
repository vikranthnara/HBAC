#!/usr/bin/env bash
# D16: Retrain L1 GRPO with parse-failure penalty in l1_schema_reward.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

PARSE_PENALTY="${PARSE_PENALTY:-0.3}"
OUT="${OUT:-checkpoints/phase3_parse_penalty}"

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

echo "D16 L1 retrain done -> ${OUT}"
