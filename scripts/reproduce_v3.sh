#!/usr/bin/env bash
# One-command V3 reproduction (oracle → analysis). GPU steps require Rivanna.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== DPO contamination audit ==="
python -m hbac.scripts.audit_dpo_contamination \
  --eval-batches checkpoints/eval_n1000/batches.jsonl \
  --output results/dpo_contamination_audit.json

echo "=== DPO holdout policy ==="
python -m hbac.scripts.prepare_dpo_holdout \
  --output results/dpo_holdout_policy.json

python -m hbac.scripts.audit_dpo_contamination \
  --eval-batches checkpoints/eval_n1000/batches.jsonl \
  --exclude-benchmarks livecodebench \
  --output results/dpo_contamination_audit_holdout.json

echo "=== Migrate n2000 artifact name ==="
python -m hbac.scripts.migrate_n2000_artifact

echo "=== D18 oracle ladder ==="
python -m hbac.scripts.analyze_d18_oracle_ladder \
  --output results/d18_oracle_ladder.json

echo "=== hard_min_frac oracle ablation ==="
python -m hbac.scripts.analyze_hard_min_frac_oracle \
  --output results/hard_min_frac_oracle_sweep.json

echo "=== Credit beta sweep ==="
python -m hbac.scripts.analyze_credit_beta_sweep \
  --batches-path checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/batches.jsonl \
  --output results/credit_beta_sweep.json

echo "=== Power analysis (n=2000 legacy) ==="
python -m hbac.scripts.power_analysis_paired \
  --output results/power_analysis_paired.json

echo "=== Paired allocator analysis (legacy n=2000) ==="
python -m hbac.scripts.analyze_paired_allocators \
  --output results/paired_allocator_analysis.json

echo "=== Lock canonical artifacts ==="
python -m hbac.scripts.lock_canonical_artifacts

echo "=== Rivanna GPU (manual) ==="
echo "  bash scripts/rivanna/submit_capability_pilot.sh"
echo "  bash scripts/rivanna/submit_v3_d18_live.sh"
echo "  sbatch slurm/train_llm_dpo_holdout.sh"
