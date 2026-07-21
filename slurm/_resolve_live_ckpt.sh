# Resolve L1/L2/batches for live compose eval (matches eval_compose_live_v2.sh).
HBAC_LIVE_TAG="${HBAC_LIVE_TAG:-bf040_seed47}"
CKPT_DIR=$(ls -td "checkpoints/variant_b/parallel_tight/${HBAC_LIVE_TAG}/stage3"/*/ 2>/dev/null | head -1)
if [[ -z "${CKPT_DIR}" ]]; then
  echo "ERROR: no checkpoint under checkpoints/variant_b/parallel_tight/${HBAC_LIVE_TAG}/stage3"
  exit 1
fi
export HBAC_CKPT_DIR="${CKPT_DIR}"
export HBAC_L2="${CKPT_DIR}/frozen_l2_controller.npz"
export HBAC_L1="${CKPT_DIR}/level1_policy.npz"
export HBAC_BATCHES="${CKPT_DIR}/batches.jsonl"
echo "Live ckpt: ${CKPT_DIR}"
