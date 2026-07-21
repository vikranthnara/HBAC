#!/bin/bash
#SBATCH -J hbac_dpo_v2
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-user=eyu8ps@virginia.edu
#SBATCH --mail-type=END,FAIL
set -euo pipefail
module purge && module load miniforge/24.3.0-py3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

HBAC_ROOT="${HBAC_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"
cd "${HBAC_ROOT}"
source slurm/_gpu_setup.sh

export HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export PYTHONUNBUFFERED=1

python -m hbac.scripts.collect_stub_oracles --output data/oracles/stub_live
python -m hbac.scripts.analyze_capabilities \
  --oracle-path data/oracles \
  --output results/capability_report.json

python -m hbac.scripts.train_llm_dpo \
  --oracle-path data/oracles \
  --model "${HBAC_LLM_MODEL}" \
  --max-pairs 600 \
  --epochs 3 \
  --sft-epochs 3 \
  --reject-modes wrong_tool \
  --output checkpoints/llm_dpo \
  --run-suffix capability_v2

RUN_DIR=$(ls -td checkpoints/llm_dpo/*_capability_v2/ | head -1)
python -m hbac.scripts.eval_grpo_format \
  --oracle-path data/oracles \
  --model "${HBAC_LLM_MODEL}" \
  --lora-path "${RUN_DIR}/model" \
  --limit 100 \
  --output results/grpo_format_dpo_v2.json

# Gate: require tool-name match before scheduling live eval
MATCH=$(python -c "import json; d=json.load(open('results/grpo_format_dpo_v2.json')); print(d.get('tool_name_match_rate',0))")
echo "DPO v2 tool_name_match_rate=${MATCH}"
if python -c "import json; d=json.load(open('results/grpo_format_dpo_v2.json')); exit(0 if d.get('tool_name_match_rate',0)>=0.3 else 1)"; then
  echo "Format gate PASSED — live eval can proceed"
else
  echo "Format gate WARN — tool_name_match below 0.3; review before live eval"
fi

echo "Phase 3c DPO v2 done $(date) -> ${RUN_DIR}"
