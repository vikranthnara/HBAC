#!/bin/bash
# Retrain DPO excluding eval benchmark families (holdout policy).
#SBATCH -J hbac_dpo_holdout
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh

# Exclude LCB (exact task-ID overlap with eval); train pairs from stub/mock oracles.
EXCLUDE="livecodebench"
ORACLE="${DPO_ORACLE_PATH:-data/oracles}"

python -m hbac.scripts.train_llm_dpo \
  --oracle-path "${ORACLE}" \
  --model "${DPO_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
  --max-pairs 600 \
  --epochs 3 \
  --sft-epochs 3 \
  --exclude-benchmarks "${EXCLUDE}" \
  --run-suffix capability_holdout

python -m hbac.scripts.audit_dpo_contamination \
  --oracle-root "${ORACLE}" \
  --eval-batches checkpoints/eval_n1000/batches.jsonl \
  --exclude-benchmarks "${EXCLUDE}" \
  --output results/dpo_contamination_audit_holdout.json

echo "DPO holdout retrain done $(date)"
