#!/usr/bin/env bash
# One-time Rivanna setup for HBAC Phase 3.
# Run on login node: bash scripts/rivanna/setup_cluster.sh
set -euo pipefail

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac}"
REPO_URL="${REPO_URL:-}"

echo "=== HBAC Rivanna setup ==="
echo "Target: ${HBAC_ROOT}"

module purge
module load miniforge/24.3.0-py3.11

if [[ ! -d "${HBAC_ROOT}" ]]; then
  mkdir -p "$(dirname "${HBAC_ROOT}")"
  if [[ -n "${REPO_URL}" ]]; then
    git clone "${REPO_URL}" "${HBAC_ROOT}"
  else
    echo "Set REPO_URL or copy repo to ${HBAC_ROOT}"
    exit 1
  fi
fi

cd "${HBAC_ROOT}"

if ! conda env list | grep -q '^hbac '; then
  conda create -n hbac python=3.11 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac
pip install -e ".[dev]"

mkdir -p data/oracles checkpoints/variant_a checkpoints/variant_a/latest checkpoints/variant_b logs results slurm

# Symlink latest L2 for Slurm scripts
if compgen -G "checkpoints/variant_a/*/stage1_stop_controller.npz" > /dev/null; then
  bash scripts/rivanna/link_latest_checkpoint.sh 2>/dev/null || true
fi

echo "=== Setup complete ==="
echo "Activate: conda activate hbac"
echo "Sync data from laptop: bash scripts/rivanna/sync_to_rivanna.sh"
