#!/bin/bash
#SBATCH -J hbac_vb_s4
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=END,FAIL
# Variant B Stage 4 joint L1+L2 — submit after parallel Stage 3 array completes.

set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
pip install -q python-dotenv -e ".[dev]" 2>/dev/null || pip install -q python-dotenv

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"
cd "${HBAC_ROOT}"

bash scripts/rivanna/link_latest_checkpoint.sh

# Pick best Stage 3 L1 checkpoint by mean_batch_reward in train_log.jsonl
BEST_DIR=$(python - <<'PY'
import json
from pathlib import Path

root = Path("checkpoints/variant_b/parallel")
best_score = float("-inf")
best_l1 = None
best_l2 = None

for log in root.rglob("stage3/*/train_log.jsonl"):
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    if not rows:
        continue
    score = rows[-1].get("mean_batch_reward", float("-inf"))
    if score > best_score:
        best_score = score
        run_dir = log.parent
        l1 = run_dir / "level1_policy.npz"
        l2 = run_dir / "frozen_l2_controller.npz"
        if l1.is_file() and l2.is_file():
            best_score = score
            best_l1 = l1
            best_l2 = l2

if best_l1 is None:
    raise SystemExit("No Stage 3 checkpoints under checkpoints/variant_b/parallel/")

print(best_l1.parent)
PY
)

echo "Best Stage 3 run: ${BEST_DIR}"

L1_CKPT="${BEST_DIR}/level1_policy.npz"
L2_CKPT="${BEST_DIR}/frozen_l2_controller.npz"

python -m hbac.scripts.train_variant_b \
  --oracle-path data/oracles \
  --checkpoint "${L2_CKPT}" \
  --stage 4 \
  --no-freeze-l2 \
  --budget-fraction 0.75 \
  --grpo-groups 16 \
  --num-batches 50 \
  --epochs 8 \
  --use-counterfactual \
  --seed 42 \
  --output checkpoints/variant_b/stage4_joint

# Final Phase 3 report + gates (Variant B track)
python -m hbac.scripts.run_phase3 \
  --oracle-path data/oracles \
  --checkpoint checkpoints/variant_a/latest \
  --output checkpoints/phase3 \
  --grpo-groups 16 \
  --num-batches 50 \
  --epochs 12 \
  --skip-variant-a

python -m hbac.scripts.check_phase3 \
  --oracle-path data/oracles \
  --phase3-path checkpoints/phase3

echo "Variant B Stage 4 + Phase 3 gates complete $(date)"
