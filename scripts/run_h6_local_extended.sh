#!/usr/bin/env bash
# Extended local H6: counterfactual credit on/off at higher batch scale (oracle replay).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

ORACLE="${ORACLE_PATH:-data/oracles}"
BATCHES="${H6_BATCHES:-50}"
EPOCHS="${H6_EPOCHS:-10}"
BF="${H6_BUDGET_FRACTION:-0.4}"
OUT_BASE="checkpoints/ablations/h6_local_ext"
RESULTS="results"

mkdir -p "${OUT_BASE}" "${RESULTS}"

run_track() {
  local tag="$1"
  shift
  local out="${OUT_BASE}/${tag}"
  echo "=== H6 local extended: ${tag} (${BATCHES} batches, ${EPOCHS} epochs) ==="
  python -m hbac.scripts.train_variant_b \
    --oracle-path "${ORACLE}" \
    --checkpoint checkpoints/variant_a \
    --stage 3 \
    --freeze-l2 \
    --budget-fraction "${BF}" \
    --num-batches "${BATCHES}" \
    --epochs "${EPOCHS}" \
    --grpo-groups 12 \
    --seed $((60 + RANDOM % 100)) \
    --output "${out}" \
    "$@"

  local run_dir
  run_dir=$(ls -td "${out}/stage3"/*/ | head -1)
  python -m hbac.scripts.eval_compose \
    --batches-path "${run_dir}/batches.jsonl" \
    --l2-checkpoint "${run_dir}/frozen_l2_controller.npz" \
    --l1-checkpoint "${run_dir}/level1_policy.npz" \
    --oracle-path "${ORACLE}" \
    --output "${RESULTS}/h6_local_ext_${tag}.json"
  echo "Wrote ${RESULTS}/h6_local_ext_${tag}.json"
}

run_track with_credit '--use-counterfactual'
run_track no_credit '--no-use-counterfactual'

python - <<'PY'
import json
from pathlib import Path

rows = []
for tag in ("with_credit", "no_credit"):
    p = Path(f"results/h6_local_ext_{tag}.json")
    if p.exists():
        d = json.loads(p.read_text())
        h = d.get("hbac_joint", {})
        u = d.get("uniform", {})
        rows.append({
            "tag": tag,
            "hbac_pass_at_1": h.get("pass_at_1"),
            "uniform_pass_at_1": u.get("pass_at_1"),
            "hbac_mean_reward": h.get("mean_batch_reward"),
            "uniform_mean_reward": u.get("mean_batch_reward"),
        })
summary = {"scale": "local_extended", "tracks": rows}
out = Path("results/h6_local_ext_summary.json")
out.write_text(json.dumps(summary, indent=2))
print(f"Wrote {out}")
for r in rows:
    print(r)
PY
