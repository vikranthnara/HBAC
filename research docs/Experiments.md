# HBAC Experiments

**Context store** for experimental setup, datasets, baselines, and evaluation protocol.  
Companion docs: [Methodology.md](Methodology.md) · [Results.md](Results.md) · [Related Work.md](Related%20Work.md)

*Last updated: July 5, 2026*

---

## 1. Experimental goals

We test Tier-C hypotheses H1–H8 (see [Research Plan.md](Research%20Plan.md) §10):

| Priority | Hypothesis | Primary metric |
|----------|------------|----------------|
| **P0** | H4: HBAC beats CLEAR/uniform under global budget | Pass@1, mean batch reward |
| **P0** | H1–H3: HBAC vs ReAct/TAB/Re-FORC | Pass@1, utility/token |
| **P1** | H6: Counterfactual credit helps L1 | L1 GRPO convergence, pass@1 |
| **P1** | H7: KL prevents stop collapse | Premature-stop rate, val accuracy |
| **P2** | H5: Draft signals help L2 | Stop accuracy |
| **P2** | Phase 3b: LoRA improves live tool JSON | Live pass@1, valid-JSON rate |
| **P3** | H8: Curriculum beats joint training | Learning curves |

---

## 2. Hardware and software

| Setting | Configuration |
|---------|---------------|
| **Cluster** | UVA Rivanna (`shen` partition), NVIDIA A6000 (48 GB) |
| **Isolated run** | `/standard/liverobotics/hbac-run-20260630T183941Z` |
| **Local dev** | macOS, Python 3.11+, pytest gates |
| **LLM inference (live)** | HuggingFace `transformers`, Qwen2.5-7B-Instruct |
| **LLM training (3b)** | TRL GRPO + PEFT LoRA (rank 16) |
| **Orchestration** | Slurm array jobs, impact feedback loop (`scripts/run_impact_loop.sh`) |

---

## 3. Datasets and environments

### 3.1 Oracle training data

| Source | Env | Role | Approx. scale |
|--------|-----|------|---------------|
| Seed oracles | `mock` | CI / fast iteration | ~50 trajectories |
| Collected oracles | `livecodebench` | Stage 1–3 training | 500–1000 target |
| Stub oracles | `tau_bench`, `toolbench`, `mock`, `swe_bench` | Live eval alignment | Per-env stubs |

Oracle format: ReAct trajectories with tool JSON, per-step budgets, success labels (`data/oracles/**/oracles.jsonl`).

### 3.2 Evaluation benchmarks

| ID | Benchmark | Mode in HBAC | Notes |
|----|-----------|--------------|-------|
| B1 | SWE-Bench Verified | Stub + Docker path | 500 instances [A6, A7] |
| B2 | LiveCodeBench | Local + oracle replay | Contamination-free [A8] |
| B3 | ToolBench | Stub | API orchestration [A19] |
| B4 | τ-bench | Stub | User interaction [A9] |
| B5 | Mock | Deterministic | CI gates |

**Live eval constraint:** LCB requires oracle replay (no live API in tight-budget eval); stub envs used for end-to-end LLM rollouts.

---

## 4. Baselines

| ID | Method | HBAC implementation | What it isolates |
|----|--------|---------------------|------------------|
| BL1 | **ReAct** [A1] | `hbac/baselines/react.py` | Fixed-budget think–act–observe |
| BL2 | **TAB** [A2] | Heuristic turn budgeting | Per-turn adaptive budgets (no global batch) |
| BL3 | **Re-FORC** [A3] | Heuristic forecaster stop | Early stop via reward forecast (no global budget) |
| BL4 | **CLEAR** [A4] | `hbac/baselines/clear.py` | Shadow-price economic allocation |
| BL5 | **Uniform** | Equal split of \(B_{\text{total}}\) | Naive batch allocator |
| BL6 | **ZEBRA** [A5] | `hbac/baselines/zebra.py` | Zero-shot water-filling (oracle-weight proxy) |

**Fidelity note:** BL2/BL3/BL6 are engineering proxies (heuristic), not paper checkpoints. Comparisons hold budget fixed externally (Tier C).

---

## 5. HBAC experimental conditions

### 5.1 Oracle replay (H4 tight budget)

| Parameter | Value |
|-----------|-------|
| Eval script | `eval_compose.py` |
| Budget fractions | 40%, 45%, 50% of oracle token sum |
| Per-task floor | `len(tasks) × 40` tokens |
| Batches per track | ~50 (up to 500 tasks) |
| L2 | Frozen stop controller |
| L1 | GRPO-trained schema policy (Variant B) |
| Seeds | bf040/seed47, bf045/seed46, bf050/seed45 |

### 5.2 Live LLM compose eval

| Parameter | Value |
|-----------|-------|
| Eval script | `eval_compose_live.py` |
| Model | Qwen2.5-7B-Instruct + DPO v2 LoRA |
| Benchmarks | `tau_bench`, `toolbench`, `mock`, `swe_bench` |
| Global budget fraction | 40% |
| Per-task floor | **600** (generous) or **400** (tight breakthrough) |
| Batches | 50 (300 tasks) |
| Allocators | Uniform, CLEAR, **ZEBRA**, HBAC joint |
| Flags | `--scarcity-boost`, `--shift-fraction`, `--swe-min-reserve` (D12), `--roi-skip` (D14) |

**Canonical artifacts:** floor=600 → `compose_live_bf040_seed47_dpo_v2.json`; floor=400 → `compose_live_bf040_floor400_dpo_v2.json`

### 5.3 Floor dose-response (P1)

| Parameter | Value |
|-----------|-------|
| Floors tested | 300, 400, 450, 500, 600 |
| Script | `slurm/eval_live_floor_sweep.sh`, `eval_live_floor_dose.sh` |
| Analysis | `analyze_floor_sweep.py` |

### 5.4 Oracle floor-matched (P3)

| Parameter | Value |
|-----------|-------|
| Script | `eval_compose_floor.py`, `eval_oracle_floor_sweep.py` |
| Purpose | Same batch construction as live; oracle replay |

### 5.5 HBAC optimizations (beyond original plan)

| ID | Implementation | Eval |
|----|----------------|------|
| D12 | `scarcity_boost_alloc` + CLI flags | `eval_live_scarcity_boost.sh`, `eval_live_scarcity_boost_refined.sh` |
| D13 | `hbac/training/metrics.py` → compliant utility | `analyze_compliant_utility.py` |
| D16 | `parse_penalty` in `l1_schema_reward` + `run_phase3 --parse-penalty` | `slurm/train_l1_parse_penalty.sh` |
| D14 | `roi_skip_alloc` + `--roi-skip` in live eval | `slurm/eval_live_roi_skip.sh` |

### 5.6 Ablation protocols

| Ablation | Script | Variables |
|----------|--------|-----------|
| H6 counterfactual credit | `train_variant_b.py` | `--use-counterfactual` vs `--no-use-counterfactual` |
| H7 KL coefficient | `ablate_kl.py` | `kl_coef ∈ {0, 0.01, 0.02, 0.05}` |
| H5 draft signals | `ablate_draft.py` | 7-dim vs 9-dim L2 features |
| Phase 3b GRPO v1 | `train_llm_grpo.py` | overlap reward, 128 samples |
| Phase 3b GRPO v2 | `train_llm_grpo_v2.py` | SFT-only vs SFT+tool-aware GRPO, 400 steps |

### 5.7 H6 long-scale (Rivanna)

| Parameter | Value |
|-----------|-------|
| `num_batches` | 150 |
| `epochs` | 12 |
| `grpo_groups` | 16 |
| `budget_fraction` | 0.4 |
| Tracks | with_credit / no_credit (array) |

---

## 6. Metrics

| Metric | Formula / definition | Reported in |
|--------|---------------------|-------------|
| **Pass@1** | \(\frac{1}{n}\sum_i S_i\) | All compose evals |
| **Mean batch reward** | \(R^{(1)}\) averaged over batches | H4, live eval |
| **Batch violation rate** | \(P(\sum_i C_i > B_{\text{total}})\) | Compose eval |
| **Allocation variance** | Var(\(b_1,\ldots,b_n\)) | HBAC vs uniform |
| **Valid JSON rate** | Fraction of generations parseable as tool JSON | `eval_grpo_format.py` |
| **Compliant utility (P2)** | \(R \times (1 - \text{violations}) - 0.5 \times \text{parse\_fail}\) | `analyze_compliant_utility.py` |
| **Reward per success per token** | pass@1 / tokens | `metrics.py` |
| **Parse failures / task** | Invalid tool JSON steps | Live compose |
| **Premature-stop rate** | Stops before env success | H7 probes |

**Primary success criteria (updated July 2026):**

1. **H4 oracle:** HBAC pass@1 > uniform at equal \(B_{\text{total}}\) — **CONFIRMED (+20 pp)**
2. **Live generous (floor=600):** HBAC Pareto-dominates on reward + tokens + violations — **CONFIRMED**
3. **Live tight (floor=400):** HBAC pass@1 > uniform — **CONFIRMED (+27.6 pp)**
4. **vs CLEAR:** HBAC compliant utility ≫ CLEAR — **CONFIRMED (~39×)**

---

## 7. Go/no-go gates

Automated via `check_go_no_go.py` and `check_phase3.py` (15 + 4 gates). All gateway gates **PASS** before Phase 3 scaling.

Key thresholds:
- Oracle yield ≥ 60% on easy/medium split
- Stop format compliance ≥ 95%
- Budget violation (L2) < 2%
- KL stability tail ∈ [0.01, 0.05]

---

## 8. Reproducibility commands

```bash
# Gates
pytest tests/ -q
python -m hbac.scripts.check_go_no_go --oracle-path data/oracles

# Oracle H4 replay
python -m hbac.scripts.eval_compose \
  --batches-path checkpoints/variant_b/.../batches.jsonl \
  --l2-checkpoint .../frozen_l2_controller.npz \
  --l1-checkpoint .../level1_policy.npz \
  --oracle-path data/oracles \
  --output results/compose_tight.json

# Live compose (canonical)
python -m hbac.scripts.eval_compose_live --batches-path ... --lora-path ...

# Floor sweep analysis
python -m hbac.scripts.analyze_floor_sweep
python -m hbac.scripts.analyze_baseline_pareto
python -m hbac.scripts.analyze_compliant_utility
python -m hbac.scripts.analyze_unified_narrative

# Oracle floor-matched (P3)
python -m hbac.scripts.eval_oracle_floor_sweep --floors "300,400,450,500,600"

# Discovery waves (Rivanna)
bash scripts/rivanna/submit_discovery_wave8.sh  # optimization-first, after pull
bash scripts/rivanna/pull_from_rivanna.sh
```

---

## 9. Experiment queue (prioritized)

See [Research Discovery.md](Research%20Discovery.md) §1 for live queue.

### In flight (July 5)

| Job | Task |
|-----|------|
| 16771391 | Floor=500 |
| 16771869 | Floors 300, 450 |
| 16771884 | ZEBRA + all baselines |
| 16772086 | D12 scarcity boost |

### Wave 8 — optimization-first (next after pull)

| Priority | ID | Task |
|----------|-----|------|
| P0 | D12 | Confirm scarcity boost vs floor=400 baseline |
| P1 | D16 | L1 retrain with parse-failure penalty |
| P2 | D14 | ROI task skipping at floor<350 |
| P3 | D12+D16 | Combined optimization eval |

### Original plan backlog (deprioritized)

| Experiment | Notes |
|------------|-------|
| TRACE capability LoRAs | Done (DPO v2) |
| Full τ/ToolBench oracles | Env expansion |
| H8 curriculum | Deferred |
| TAB live baseline | Orthogonal axis |
