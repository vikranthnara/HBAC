#!/usr/bin/env bash
# Submit full-scale Variant B hierarchical GRPO on Rivanna (run on LOGIN node).
# Usage: bash scripts/rivanna/submit_fullscale_variant_b.sh
set -euo pipefail

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"
cd "${HBAC_ROOT}"

mkdir -p logs checkpoints/variant_b/parallel results

echo "=== HBAC full-scale Variant B submission ==="
echo "Root: ${HBAC_ROOT}"
echo "Host: $(hostname)"
echo "Date: $(date)"

# Preflight
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
python -c "import hbac; from hbac.training.controller import MonolithicController; from pathlib import Path; c=sorted(Path('checkpoints/variant_a').rglob('stage1_stop_controller.npz')); assert c, 'missing L2 checkpoint'; MonolithicController.load(c[-1]); print('preflight OK')"

SMOKE_ID=""
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo "Submitting GPU smoke test..."
  SMOKE_ID=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}" slurm/smoke_gpu.sh)
  echo "  smoke job: ${SMOKE_ID}"
fi

echo "Submitting parallel Variant B Stage 3 array (3 budget tracks)..."
ARRAY_ID=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}" slurm/variant_b_parallel_array.sh)
echo "  array job: ${ARRAY_ID}"

if [[ "${SUBMIT_GPU_FULLSCALE:-1}" == "1" ]]; then
  echo "Submitting GPU full-scale run_phase3 (gpu partition)..."
  GPU_ID=$(sbatch --parsable --export=ALL,HBAC_ROOT="${HBAC_ROOT}" slurm/variant_b_fullscale_gpu.sh)
  echo "  gpu job: ${GPU_ID}"
else
  GPU_ID=""
fi

STAGE4_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_ID} --export=ALL,HBAC_ROOT="${HBAC_ROOT}" slurm/variant_b_stage4_joint.sh)
echo "  stage4 job: ${STAGE4_ID} (depends on array ${ARRAY_ID})"

cat > logs/fullscale_variant_b_jobids.txt <<EOF
smoke=${SMOKE_ID}
parallel_array=${ARRAY_ID}
gpu_fullscale=${GPU_ID}
stage4=${STAGE4_ID}
hbac_root=${HBAC_ROOT:-$(pwd)}
submitted=$(date -Iseconds)
EOF

echo ""
echo "=== Submitted ==="
echo "  Parallel Stage 3:  squeue -j ${ARRAY_ID}"
echo "  Stage 4 + gates:     squeue -j ${STAGE4_ID}"
echo "  Monitor:             squeue -u \$USER"
echo "  Job IDs saved:       logs/fullscale_variant_b_jobids.txt"
