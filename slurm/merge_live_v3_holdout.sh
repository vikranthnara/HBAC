#!/bin/bash
# Merge holdout-LoRA V3 live shards + paired analysis.
#SBATCH -J hbac_v3_hmerge
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

if [[ ! -f results/rivanna/v3_live_holdout_shards/hbac_fair.json ]] \
  && [[ -f results/rivanna/v3_live_holdout_shards/hbac_guardrail.json ]]; then
  cp results/rivanna/v3_live_holdout_shards/hbac_guardrail.json \
    results/rivanna/v3_live_holdout_shards/hbac_fair.json
fi

python -m hbac.scripts.merge_compose_live_shards \
  --shard-dir results/rivanna/v3_live_holdout_shards \
  --output results/rivanna/compose_live_v3_holdout_floor400_n2000.json \
  --keys "hbac_d18,type_prior,hbac_joint,hbac_guardrail,uniform,hbac_fair" \
  --allow-partial \
  --v3

python -m hbac.scripts.analyze_paired_allocators \
  --shard-dir results/rivanna/v3_live_holdout_shards \
  --merged results/rivanna/compose_live_v3_holdout_floor400_n2000.json \
  --pairs "hbac_d18:type_prior,hbac_guardrail:type_prior,hbac_joint:type_prior" \
  --output results/paired_allocator_analysis_v3_holdout.json

python -m hbac.scripts.analyze_v3_d18_live \
  --result-path results/rivanna/compose_live_v3_holdout_floor400_n2000.json \
  --paired-path results/paired_allocator_analysis_v3_holdout.json \
  --output results/v3_holdout_live_analysis.json || true

python -m hbac.scripts.lock_canonical_artifacts

echo "Holdout live merge done $(date)"
