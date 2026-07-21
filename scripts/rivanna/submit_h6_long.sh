#!/usr/bin/env bash
# H6 at scale: parallel Variant B train with vs without counterfactual credit.
set -euo pipefail

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"

module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

mkdir -p logs checkpoints/ablations/h6_long

echo "=== H6 long-scale ablation submit ==="
H6_ID=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}" slurm/variant_b_h6_long.sh)
echo "h6_long_array=${H6_ID}"
echo "h6_long_array=${H6_ID}" >> logs/impact_wave_jobids.txt
echo "submitted=$(date -u +%Y%m%dT%H%M%SZ)" >> logs/impact_wave_jobids.txt
