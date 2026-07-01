#!/bin/bash
#SBATCH -J hbac_variants
#SBATCH -p standard_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module purge
module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

cd "${HBAC_ROOT:-/standard/liverobotics/hbac}"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  python -m hbac.scripts.train_variant_b \
    --stage 3 --freeze-l2 --grpo-groups 16 \
    --num-batches 30 --epochs 10 \
    --output checkpoints/variant_b/stage3
else
  python -m hbac.scripts.train_variant_a_l1 \
    --num-batches 30 --epochs 10 \
    --output checkpoints/variant_a_l1
fi

echo "Array task ${SLURM_ARRAY_TASK_ID} done $(date)"
