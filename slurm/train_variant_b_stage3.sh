#!/bin/bash
#SBATCH -J hbac_phase3
#SBATCH -p standard_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
cd "${HBAC_ROOT:-/standard/liverobotics/hbac}"
bash scripts/rivanna/link_latest_checkpoint.sh

python -m hbac.scripts.run_phase3 \
  --oracle-path data/oracles \
  --checkpoint checkpoints/variant_a/latest \
  --output checkpoints/phase3 \
  --grpo-groups 16 \
  --num-batches 30 \
  --epochs 8

python -m hbac.scripts.check_phase3 \
  --oracle-path data/oracles \
  --phase3-path checkpoints/phase3

echo "Phase 3 complete $(date)"
