#!/usr/bin/env bash
# Phase 3b v2 wave: tool-aware SFT+GRPO train (2 tracks) + chained live eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"

LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ARCHIVE="/tmp/hbac_grpo_v2_sync.tar.gz"

echo "=== GRPO v2 improvement wave ==="
tar czf "${ARCHIVE}" \
  --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude '.env' \
  -C "${LOCAL_ROOT}" hbac scripts slurm pyproject.toml

rivanna_scp "${ARCHIVE}" "${RIVANNA_HOST}:${REMOTE_ROOT}/hbac_grpo_v2_sync.tar.gz"
rivanna_ssh bash -s <<REMOTE
set -euo pipefail
cd '${REMOTE_ROOT}'
tar xzf hbac_grpo_v2_sync.tar.gz
chmod +x slurm/*.sh scripts/rivanna/*.sh 2>/dev/null || true
export HBAC_ROOT='${REMOTE_ROOT}'
export HBAC_LLM_MODEL='Qwen/Qwen2.5-7B-Instruct'
module purge && module load miniforge/24.3.0-py3.11
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

TRAIN_ID=\$(sbatch --parsable --export=ALL,HBAC_ROOT,HBAC_LLM_MODEL slurm/train_llm_grpo_v2.sh)
LIVE_ID=\$(sbatch --parsable --dependency=afterok:\${TRAIN_ID} --export=ALL,HBAC_ROOT,HBAC_LLM_MODEL slurm/eval_compose_live_v2.sh)
echo "train_v2=\${TRAIN_ID}" | tee logs/grpo_v2_jobids.txt
echo "live_v2=\${LIVE_ID}" | tee -a logs/grpo_v2_jobids.txt
echo "submitted=\$(date -u +%Y%m%dT%H%M%SZ)" >> logs/grpo_v2_jobids.txt
cat logs/grpo_v2_jobids.txt
REMOTE

echo "Monitor: source scripts/rivanna/_ssh_env.sh && rivanna_ssh 'squeue -u eyu8ps'"
