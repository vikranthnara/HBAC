#!/bin/bash
# Merge compose-live n2000 shards after array job completes.
#SBATCH -J hbac_v3_live_merge
#SBATCH -p standard
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"

SHARD_DIR="results/live_n2000_shards"
OUTPUT="results/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json"

python -m hbac.scripts.merge_compose_live_shards \
  --shard-dir "${SHARD_DIR}" \
  --meta-path "${SHARD_DIR}/meta.json" \
  --output "${OUTPUT}"

echo "Merged live n2000 report -> ${OUTPUT} $(date)"
