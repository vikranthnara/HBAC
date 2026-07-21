#!/bin/bash
# V3: Oracle eval on real LCB + SWE Lite pool with Tier-A official baselines.
#SBATCH -J hbac_v3_oracle
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"

if [[ ! -d data/oracles/real_eval/latest ]]; then
  python -m hbac.scripts.build_real_eval_pool \
    --lcb-problems 500 \
    --swe-limit 50 \
    --output data/oracles/real_eval
fi

python -m hbac.scripts.generate_large_eval_batches \
  --oracle-path data/oracles/real_eval/latest \
  --num-batches 50 \
  --seed 47 \
  --output checkpoints/eval_real/batches.jsonl

L1="checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz"
L2="checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/frozen_l2_controller.npz"

python -m hbac.scripts.eval_compose_v2 \
  --batches-path checkpoints/eval_real/batches.jsonl \
  --l1-checkpoint "${L1}" \
  --l2-checkpoint "${L2}" \
  --oracle-path data/oracles/real_eval/latest \
  --output results/v3_real_oracle_matrix.json

echo "V3 oracle real pool done $(date)"
