#!/bin/bash
# Merge V3 live d18 shards + run paired analysis.
#SBATCH -J hbac_v3_merge
#SBATCH -p standard
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"

# Back-compat with merge script requiring legacy hbac_fair key.
if [[ ! -f results/rivanna/v3_live_d18_shards/hbac_fair.json ]] \
  && [[ -f results/rivanna/v3_live_d18_shards/hbac_guardrail.json ]]; then
  cp results/rivanna/v3_live_d18_shards/hbac_guardrail.json \
    results/rivanna/v3_live_d18_shards/hbac_fair.json
fi

python -m hbac.scripts.merge_compose_live_shards \
  --shard-dir results/rivanna/v3_live_d18_shards \
  --output results/rivanna/compose_live_v3_d18_floor400_n2000.json \
  --v3

python -m hbac.scripts.analyze_paired_allocators \
  --shard-dir results/rivanna/v3_live_d18_shards \
  --merged results/rivanna/compose_live_v3_d18_floor400_n2000.json \
  --pairs "hbac_d18:type_prior,hbac_guardrail:type_prior,hbac_joint:type_prior" \
  --output results/paired_allocator_analysis_v3_d18.json

python -m hbac.scripts.lock_canonical_artifacts

echo "V3 merge done $(date)"
