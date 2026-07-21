#!/bin/bash
#SBATCH -J hbac_h6_long
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"
cd "${HBAC_ROOT}"

TAGS=(with_credit no_credit)
FLAGS=('--use-counterfactual' '--no-use-counterfactual')
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
OUT="checkpoints/ablations/h6_long/${TAG}"

bash scripts/rivanna/link_latest_checkpoint.sh checkpoints/variant_a

python -m hbac.scripts.train_variant_b \
  --oracle-path data/oracles \
  --checkpoint checkpoints/variant_a \
  --stage 3 \
  --freeze-l2 \
  --budget-fraction 0.4 \
  --num-batches 150 \
  --epochs 12 \
  --grpo-groups 16 \
  --seed $((50 + SLURM_ARRAY_TASK_ID)) \
  --output "${OUT}" \
  "${FLAGS[$SLURM_ARRAY_TASK_ID]}"

RUN_DIR=$(ls -td "${OUT}/stage3"/*/ | head -1)
python -m hbac.scripts.eval_compose \
  --batches-path "${RUN_DIR}/batches.jsonl" \
  --l2-checkpoint "${RUN_DIR}/frozen_l2_controller.npz" \
  --l1-checkpoint "${RUN_DIR}/level1_policy.npz" \
  --oracle-path data/oracles \
  --output "results/h6_long_${TAG}.json"

python - <<'PY'
import json
from pathlib import Path

rows = []
for tag in ("with_credit", "no_credit"):
    p = Path(f"results/h6_long_{tag}.json")
    if p.exists():
        d = json.loads(p.read_text())
        rows.append({"tag": tag, "hbac_joint": d.get("hbac_joint"), "uniform": d.get("uniform")})
Path("results/h6_long_summary.json").write_text(json.dumps({"tracks": rows}, indent=2))
print("Wrote results/h6_long_summary.json")
PY

echo "H6 long ${TAG} done $(date)"
