#!/bin/bash
#SBATCH -J hbac_vb_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=END,FAIL
# Full-scale Variant B Stage 3 on GPU partition (larger batches, longer run).

set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
pip install -q python-dotenv -e ".[dev]" 2>/dev/null || pip install -q python-dotenv

cd "${HBAC_ROOT:-/standard/liverobotics/hbac}"
export PYTHONUNBUFFERED=1

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

echo "GPU full-scale Phase 3 complete $(date)"
