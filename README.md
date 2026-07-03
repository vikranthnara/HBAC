# HBAC

Hierarchical Budgeted Agent Controller (HBAC): evaluation substrate for SWE-Bench Verified and LiveCodeBench with ReAct, TAB, and Re-FORC baselines.

**Research foundation:** All design choices, citations, and testable hypotheses are documented in [Research Plan.md](Research%20Plan.md) (Tier A/B/C epistemology, bibliography [A1]–[A24], implementation fidelity §9.1). Machine-readable BibTeX: [references.bib](references.bib).

## Setup

Requires Python 3.11+, Docker (for full SWE-Bench evaluation), and API keys for cloud models.

```bash
pip install -e ".[dev]"

# Option A: FreeLLMAPI via paradocs (recommended)
python -m hbac.scripts.sync_freellmapi \
  --dir /path/to/paradocs/.freellmapi

# Option B: direct OpenAI
cp .env.example .env   # then set OPENAI_API_KEY
```

Start FreeLLMAPI first (from paradocs: `./scripts/run-freellmapi-npm.sh`).

Environment variables (loaded automatically from `.env`):

- `HBAC_FREELLMAPI_DIR` — path to `.freellmapi`; unified key read from SQLite
- `FREELLMAPI_BASE_URL` / `FREELLMAPI_API_KEY` — OpenAI-compatible proxy (default `http://127.0.0.1:3001/v1`)
- `HBAC_LLM_PROVIDER` — `auto` (default), `freellmapi`, `openai`, `anthropic`, or `vllm`
- `HBAC_LLM_MODEL` — `auto` routes via FreeLLMAPI; or a specific model id
- `OPENAI_API_KEY` — used when provider is `openai`
- `ANTHROPIC_API_KEY` — used when provider is `anthropic`

## Quick Start (Mock Env, No API)

Run unit tests:

```bash
pytest tests/ -q
```

## Run Baselines

```bash
# ReAct on mock env (requires API key for real LLM)
python -m hbac.scripts.run_baseline --baseline react --env mock --model openai:gpt-4o-mini --limit 1

# TAB with heuristic per-turn budgeting
python -m hbac.scripts.run_baseline --baseline tab --env livecodebench --local-mode --limit 2

# Re-FORC early stopping
python -m hbac.scripts.run_baseline --baseline ref_orc --env livecodebench --local-mode --limit 2

# SWE-Bench (local fallback without dataset)
python -m hbac.scripts.run_baseline --baseline react --env swe_bench --local-mode --limit 1
```

## Collect Oracle Trajectories

Successful strong-model ReAct rollouts for SFT/GRPO initialization:

```bash
python -m hbac.scripts.collect_oracles \
  --env livecodebench \
  --model openai:gpt-4o \
  --local-mode \
  --limit 10 \
  --output data/oracles/
```

## Export Training Data

```bash
python -m hbac.scripts.export_sft \
  --input-path data/oracles/livecodebench/<run_id>/oracles.jsonl \
  --output-dir data/training/ \
  --format both
```

Produces:
- `sft.jsonl` — message/label pairs with budget allocation and stop labels
- `grpo_groups.jsonl` — grouped trajectories with rewards for GRPO warm-start

## Package Layout

```
hbac/
  core/       # AgentEnv protocol, LLM backends, metrics, trajectories
  envs/       # SWE-Bench, LiveCodeBench, mock environments
  baselines/  # ReAct, TAB, Re-FORC runners
  scripts/    # CLI entry points
```

## Benchmarks

| Env | Description |
|-----|-------------|
| `swe_bench` | SWE-Bench Verified — bash/edit/submit in workspace |
| `livecodebench` | Code generation + self-repair turns |
| `mock` | Lightweight env for tests |

## Baselines

| Baseline | Description |
|----------|-------------|
| `react` | Fixed-budget Thought-Action-Observation loop |
| `tab` | Turn-adaptive token budgeting (heuristic or learned stub) |
| `ref_orc` | Re-FORC reward forecaster early stopping |

## Metrics

Each run writes `metrics.json` with Pass@1, budget violation rate, mean tokens, and utility per token.

---

## Phase 1 (Complete)

Infrastructure for evaluation and oracle collection:

| Deliverable | Status |
|-------------|--------|
| `AgentEnv` protocol + SWE-Bench / LiveCodeBench wrappers | Done |
| ReAct, TAB, Re-FORC baselines | Done |
| Oracle collection + SFT/GRPO export | Done |
| Seed oracle pipeline (no API keys) | Done |
| Phase 1 acceptance tests | Done |

```bash
# Verify Phase 1
pytest tests/test_phase1_acceptance.py -v

# Generate seed training data locally
python -m hbac.scripts.seed_oracles
```

---

## Phase 2 (Component Prototyping)

Train Variant A monolithic stop controller on seed + collected LCB oracles.

```bash
# 1. Validate reward (sweep + JSON report)
python -m hbac.scripts.validate_reward

# 2. Seed oracles (CI) + collect real LCB oracles via FreeLLMAPI
python -m hbac.scripts.seed_oracles
python -m hbac.scripts.collect_oracles \
  --env livecodebench --limit 20 \
  --output data/oracles/lcb

# 3. Train Stage-1 stop controller (PPO + reference-policy KL)
python -m hbac.scripts.train_variant_a \
  --oracle-path data/oracles \
  --subset-limit 50 \
  --kl-coef 0.02

# 4. KL ablation (H7)
python -m hbac.scripts.ablate_kl --oracle-path data/oracles

# 5. Evaluate checkpoint
python -m hbac.scripts.eval_variant_a \
  --checkpoint checkpoints/variant_a/<run_id>/stage1_stop_controller.npz \
  --oracle-path data/oracles
```

Phase 2 modules in `hbac/training/`:

- `reward.py` — \(R^{(2)} = S_i - \lambda C_i - \gamma L_i - \delta R_i\)
- `validation.py` — 5 invariants + hyperparameter sweep
- `controller.py` — Variant A monolithic stop head + frozen ref snapshot
- `ppo.py` — PPO with KL(ref ‖ new) to frozen reference policy [A11]
- `probes.py` — early-stop hacking probes for H7
- `dataset.py` — merged oracle loading, rich observations, train/val split

```bash
pytest tests/test_phase2_acceptance.py -v
```

## Go/No-Go Gates (before Phase 3)

Empirical milestones enforced by automated gates — see [Research Plan.md](Research%20Plan.md) §16.

```bash
python -m hbac.scripts.check_go_no_go --oracle-path data/oracles
# Report: results/go_no_go.json
# Exit 0 = all phases ready; exit 2 = blocked/failed gates
```

Key blockers until scaled collection: **500+ oracles**, **100-sample baseline repro**, **Level-1 allocator**, **draft NetGain**.

All 15 gates currently **PASS** — see `results/go_no_go.json`.

## Phase 3 Training (Rivanna)

Full pipeline (Stages 3→4 + Variant A + optional LLM GRPO):

```bash
# Complete Phase 3 locally or on Rivanna
python -m hbac.scripts.run_phase3 \
  --oracle-path data/oracles \
  --checkpoint checkpoints/variant_a \
  --output checkpoints/phase3 \
  --grpo-groups 16 --num-batches 30 --epochs 8

# Verify Phase 3 completion gates (4 gates: report, L1 vs uniform, pass@1, LLM GRPO)
python -m hbac.scripts.check_phase3 --phase3-path checkpoints/phase3

# Phase 3b LLM GRPO (local: gpt2 + SFT fallback; Rivanna: CUDA + TRL GRPO)
python -m hbac.scripts.train_llm_grpo --model gpt2 --max-samples 16
```

Stage 3 trains Level-1 with frozen Level-2; Variant B uses GRPO + counterfactual credit; Variant A uses utility network.

### Rivanna cluster (full-scale Variant B)

```bash
# Laptop: package + upload (password auth OK)
bash scripts/rivanna/package_for_upload.sh
scp /tmp/hbac_deploy.tar.gz eyu8ps@login.hpc.virginia.edu:/standard/liverobotics/

# Rivanna login node: extract, setup, submit parallel Variant B GRPO
mkdir -p /standard/liverobotics/hbac
tar xzf /standard/liverobotics/hbac_deploy.tar.gz -C /standard/liverobotics/hbac
cd /standard/liverobotics/hbac
bash scripts/rivanna/on_cluster_setup_and_submit.sh
```

This submits:
- `slurm/variant_b_parallel_array.sh` — **3 parallel** Stage-3 GRPO tracks (90/75/60% budget)
- `slurm/variant_b_stage4_joint.sh` — joint L1+L2 after array completes
- Phase 3 completion gates on cluster

Monitor: `squeue -u $USER` · logs in `logs/fullscale_variant_b_jobids.txt`

Or rsync instead of tarball: `bash scripts/rivanna/sync_to_rivanna.sh`

Shared lab path: `/standard/liverobotics/hbac`. Monitor SUs: `allocations -a lia-lab-members`.

Phase 3b LLM GRPO requires GPU extras: `pip install -e ".[gpu]"`.

### Results snapshot (July 2026)

Highlights from Rivanna full-scale runs (`hbac-run-20260630T183941Z`):

| Eval | HBAC joint | Uniform/CLEAR | Notes |
|------|------------|---------------|-------|
| Oracle H4 tight 40% (n=500) | **80%** pass@1, R=0.94 | **60%** | Retrained; budgets now differ by track |
| Oracle H4 tight 45% (n=500) | **80%** pass@1, R=0.98 | **60%** | `compose_tight_bf045_seed46.json` |
| Oracle H4 tight 50% (n=500) | **80%** pass@1, R=1.02 | **60%** | CLEAR violations at 50% only |
| Live LLM 7B (n=300, 40% budget) | 44.3% pass@1, **R=14.7** | 44.3%, R=0.44 / −0.27 | HBAC wins on reward + allocation |
| LLM GRPO (Phase 3b) | TRL GRPO on Qwen2.5-7B | — | `checkpoints/llm_grpo/20260703T080820Z` |
| H6 counterfactual credit | Same oracle H4 outcome | — | Credit on/off tie on quick train |
| H7 KL ablation | PASS | — | `results/kl_ablation_h7.json` |

```bash
# Audit budget_fraction in training batches
python -m hbac.scripts.audit_budget_fraction --batches-path checkpoints/.../batches.jsonl

# Rivanna next steps (tight retrain + live eval + LLM GRPO)
bash scripts/rivanna/submit_next_steps.sh

# Live eval on retrained checkpoint
HBAC_LIVE_TAG=bf040_seed47 HBAC_LIVE_SUFFIX=retrain bash scripts/rivanna/submit_live_eval.sh
```

