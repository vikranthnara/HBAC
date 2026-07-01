#!/bin/bash
#SBATCH -J hbac_smoke
#SBATCH -p interactive
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=FAIL

set -euo pipefail
module purge
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
pip install -q python-dotenv -e ".[dev]" 2>/dev/null || pip install -q python-dotenv

cd "${HBAC_ROOT:-/standard/liverobotics/hbac}"
export PYTHONPATH=.

echo "Host: $(hostname)"
echo "Date: $(date)"

python -c "
import hbac
from hbac.training.controller import MonolithicController
from pathlib import Path
ckpts = sorted(Path('checkpoints/variant_a').rglob('stage1_stop_controller.npz'), key=lambda p: p.stat().st_mtime)
assert ckpts, 'no L2 checkpoint'
MonolithicController.load(ckpts[-1])
print('HBAC smoke OK', hbac.__version__ if hasattr(hbac,'__version__') else '0.1.0')
"

python -m hbac.scripts.train_variant_b \
  --oracle-path data/oracles \
  --checkpoint checkpoints/variant_a \
  --stage 3 \
  --freeze-l2 \
  --num-batches 2 \
  --epochs 1 \
  --grpo-groups 4 \
  --output checkpoints/variant_b/smoke

echo "Smoke finished $(date)"
