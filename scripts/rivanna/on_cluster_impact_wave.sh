#!/usr/bin/env bash
# On Rivanna login node: sync latest code into isolated run dir and submit impact wave.
# Usage:
#   cd /standard/liverobotics/hbac-run-20260630T183941Z
#   bash scripts/rivanna/on_cluster_impact_wave.sh
set -euo pipefail

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
SRC="${HBAC_SRC:-/standard/liverobotics/hbac}"

echo "=== HBAC impact wave on cluster ==="
echo "Run dir: ${HBAC_ROOT}"
echo "Source:  ${SRC}"

module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

mkdir -p "${HBAC_ROOT}"
# Sync scripts + hbac package (preserve checkpoints/results on run dir)
rsync -a --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude '.env' \
  --exclude 'checkpoints' --exclude 'results' --exclude 'data' \
  "${SRC}/" "${HBAC_ROOT}/"

cd "${HBAC_ROOT}"
bash scripts/rivanna/link_latest_checkpoint.sh checkpoints/variant_a 2>/dev/null || true

export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
bash scripts/rivanna/submit_impact_wave.sh

echo ""
echo "Monitor: squeue -u \$USER"
echo "Pull (laptop): bash scripts/rivanna/pull_from_rivanna.sh"
