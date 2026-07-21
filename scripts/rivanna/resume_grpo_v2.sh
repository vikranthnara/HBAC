#!/usr/bin/env bash
# Resume GRPO v2: resubmit failed sft_only + fix live-eval dependency.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_ssh_env.sh"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

tar czf /tmp/hbac_grpo_fix.tar.gz \
  -C "${LOCAL_ROOT}" slurm/train_llm_grpo_v2.sh slurm/eval_compose_live_v2.sh hbac/training/llm_grpo_trainer.py

rivanna_scp /tmp/hbac_grpo_fix.tar.gz "${RIVANNA_HOST}:${REMOTE_ROOT}/hbac_grpo_fix.tar.gz"
rivanna_ssh bash -s <<REMOTE
set -euo pipefail
cd '${REMOTE_ROOT}'
tar xzf hbac_grpo_fix.tar.gz
export HBAC_ROOT='${REMOTE_ROOT}'
export HBAC_LLM_MODEL='Qwen/Qwen2.5-7B-Instruct'
module purge && module load miniforge/24.3.0-py3.11
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbac

# Cancel stuck live eval (array 16758055_0 failed → afterok never fires)
scancel 16758056 2>/dev/null || true

# Resubmit sft_only track only (array task 0), one at a time
SFT_ID=\$(sbatch --parsable --array=0%1 --export=ALL,HBAC_ROOT,HBAC_LLM_MODEL slurm/train_llm_grpo_v2.sh)
echo "resubmit_sft_only=\${SFT_ID}"

# Live eval: wait for running sft_grpo (16758055_1) if still active, else any completed v2 ckpt
GRPO_JOB=\$(sacct -j 16758055_1 --format=JobID,State -n 2>/dev/null | head -1 | awk '{print \$2}')
if [[ "\${GRPO_JOB}" == "RUNNING" ]]; then
  LIVE_GRPO=\$(sbatch --parsable --dependency=afterok:16758055_1 --array=1%1 \
    --export=ALL,HBAC_ROOT,HBAC_LLM_MODEL slurm/eval_compose_live_v2.sh)
  echo "live_sft_grpo=\${LIVE_GRPO} (after 16758055_1)"
fi
LIVE_SFT=\$(sbatch --parsable --dependency=afterok:\${SFT_ID} --array=0%1 \
  --export=ALL,HBAC_ROOT,HBAC_LLM_MODEL slurm/eval_compose_live_v2.sh)
echo "live_sft_only=\${LIVE_SFT} (after \${SFT_ID})"

{
  echo "resubmit_sft_only=\${SFT_ID}"
  echo "live_sft_only=\${LIVE_SFT}"
  echo "live_sft_grpo=\${LIVE_GRPO:-pending_16758055_1}"
  echo "resumed=\$(date -u +%Y%m%dT%H%M%SZ)"
} >> logs/grpo_v2_jobids.txt
squeue -u eyu8ps
REMOTE
