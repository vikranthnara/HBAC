# Shared GPU job setup: conda env + HF/transformers (no vLLM compile on Rivanna).
# Source from slurm/eval_compose_live.sh and slurm/train_llm_grpo.sh.

HBAC_LLM_MODEL="${HBAC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export HF_HOME="${HF_HOME:-${HBAC_ROOT:-/tmp}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
mkdir -p "${HF_HOME}"

if ! python -c "import hbac" >/dev/null 2>&1; then
  pip install -q python-dotenv -e .
else
  pip install -q python-dotenv 2>/dev/null || true
fi
pip install -q "torch>=2.2.0" "transformers>=4.40.0" "accelerate>=0.30.0" "peft>=0.11.0" "trl>=0.9.0" 2>/dev/null || true

python -c "import torch; import transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi -L 2>/dev/null || echo "WARN: nvidia-smi unavailable"
