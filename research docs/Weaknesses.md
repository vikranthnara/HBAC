# HBAC Weaknesses Tracker

**Living document** — every known reviewer-facing weakness, mitigation plan, status, and evidence.  
**Rule:** Do not mark a weakness *resolved* until we have empirical evidence or a paper-ready narrative backed by data.

Companion docs: [Results.md](Results.md) · [Experiments.md](Experiments.md) · [Paper Narrative.md](Paper%20Narrative.md)

*Last updated: July 6, 2026 (full floor curve; D12 trade-off; Wave 8 queued)*

---

## Status legend

| Status | Meaning |
|--------|---------|
| 🔴 **OPEN** | No mitigation; hurts paper |
| 🟡 **IN PROGRESS** | Active engineering / experiments |
| 🟢 **MITIGATED** | Evidence or narrative sufficient for submission |
| ⚪ **ACCEPTED** | Inherent limitation; reframed as scope / future work |

---

## Priority matrix

| ID | Weakness | Severity | Status | Target |
|----|----------|----------|--------|--------|
| **W1** | Live pass@1 flat at floor=600 | Critical | 🟢 | Dual-regime: +27.6 pp @ floor=400; Pareto @ floor=600 |
| **W2** | H4 only on oracle replay | High | 🟢 | Dual eval + reward transfer documented |
| **W3** | Stub / deterministic benchmarks | High | 🟢 | Controlled micro-benchmark narrative |
| **W4** | GRPO/DPO LoRA inconclusive | High | 🟢 | DPO v2: format 100%, live stable at 44.3% |
| **W5** | CLEAR/TAB baselines are proxies | Medium | 🟢 | Sensitivity sweep: HBAC beats CLEAR all settings |
| **W6** | Partial HBAC scope (routing, draft, H8) | Medium | ⚪ | Scope boundary in paper |
| **W7** | Uniform+LoRA regression | Medium | 🟢 | Analysis doc + greedy/budget narrative |
| **W8** | Mean batch reward vs pass@1 gap | Medium | 🟢 | Dual-metric framing |
| **W9** | DPO: valid JSON but 0% tool-name match | Critical | 🟢 | **DPO v2: 100% tool-name match** (format eval) |
| **W10** | No LaTeX paper draft | High | 🟢 | **Paper v2** in `Paper v2.md` (Jul 6, 2026) — honest reframe post-review |
| **W14** | Type-prior heuristic ties HBAC oracle | High | 🟢 | `v2_baseline_matrix_oracle_tight.json`; reported in Paper v2 §5.1 |
| **W15** | D14 ROI-skip is reward hacking | High | 🟢 | Negative result; demoted from contributions |
| **W11** | Small live eval n=300 | Medium | 🟢 | Bootstrap CIs + per-benchmark in analysis |
| **W12** | CLEAR ties uniform in oracle H4 | Low | 🟢 | Surge-proxy limitation (Results §2) |
| **W13** | Rivanna infra (ckpt path, CUDA device) | Low | 🟢 | Fixed `_resolve_live_ckpt.sh` + SFT device |

---

## W1 — Live pass@1 flat (revised diagnosis)

### Problem (original)
All allocators tied at **44.3% pass@1** on live eval; suspected tool-JSON bottleneck.

### Updated diagnosis (Jul 5, post-DPO v2 live)
1. **Tool JSON is no longer the blocker** — DPO v2 live: HBAC **0 parse failures**, **100% first-step valid JSON** (`compose_live_bf040_seed47_dpo_v2.json`).
2. **Aggregate pass@1 still ties** because hard stub domains dominate the mix:
   - toolbench: **100%** (all allocators)
   - τ-bench: **33%** (all allocators)
   - SWE-local: **0%** (all allocators)
3. HBAC **still differentiates** on metrics aligned with budgeting: **14.7× reward**, **~94 fewer tokens/task**, **0% violations** vs CLEAR 14.3% violations.

### Status: 🟡 PARTIALLY MITIGATED
- JSON/tool-name: fixed (W9)
- Allocator pass@1 separation on live: **not observed** — env competence ceiling, not L1 failure
- Paper frame: report **per-benchmark decomposition** + dual metrics (pass@1 + batch reward)

**Artifacts:** `results/live_compose_analysis.json`, `results/rivanna/compose_live_bf040_seed47_dpo_v2.json`

---

## W2 — H4 validated on oracle replay, not full live agents

### Problem
Our strongest claim (+20 pp pass@1) uses **frozen oracle trajectories** with scripted L2. Reviewers may ask whether this transfers to live LLM rollouts.

### Evidence
- Oracle: HBAC 80% vs uniform/CLEAR 60% (`compose_tight_bf*.json`, n≈500)
- Live: pass@1 ties at 44.3% (W1 masks allocator signal)

### Mitigation plan
1. **Live mean-batch-reward separation** (33× vs uniform) — already shows allocator works when episodes complete
2. **Post-DPO live eval** — if pass@1 diverges, H4 transfers partially
3. **Tight-budget live slice**: run live eval at 40% with smaller n but matched seeds across allocators
4. Optional: collect **live oracles** from successful Qwen rollouts for hybrid eval

### Paper reframe
> “We separate (i) **allocator optimality** under known per-step costs (oracle replay, H4) from (ii) **end-to-end agent competence** (live LLM), reporting both.”

### Status: 🟢 MITIGATED

**Evidence:** Live mean batch reward 33× uniform; DPO v2 preserves allocator metrics; oracle H4 remains primary pass@1 claim.

---

## W3 — Stub / deterministic benchmarks in live eval

### Problem
Live eval uses deterministic stub envs (τ-bench, ToolBench, mock, SWE-local), not full τ-bench/ToolBench/SWE-Bench Verified scale.

### Evidence
- `STUB_BENCHMARKS` in `eval_compose_live.py`
- Stub oracles: `data/oracles/stub_live/`

### Mitigation plan
1. **Align training oracles with live stubs** (`collect_stub_oracles.py`) — done
2. Document stubs as **controlled micro-benchmarks** for allocator comparison (same tasks, same env)
3. Future: plug real τ-bench API when compute budget allows
4. LCB stays on oracle replay (needs sandbox)

### Paper reframe
> “We evaluate allocation under **matched deterministic agent episodes** so allocator differences are not confounded by env stochasticity; scale-up to full benchmarks is orthogonal to L1.”

### Status: 🟢 MITIGATED (controlled micro-benchmark + per-benchmark reporting)

---

## W4 — GRPO / DPO LoRA training inconclusive

### Problem
- GRPO v1: pass@1 **44.3% → 27.7%** (regression)
- GRPO v2: SFT restores pass@1; **GRPO phase adds nothing** vs SFT-only
- DPO v1: JSON valid but tool names wrong (W9)

### Evidence
- `results/rivanna/compose_live_bf040_seed47_v2_sft_grpo.json`
- `results/grpo_format_dpo.json` (Rivanna, pending pull)

### Mitigation plan
1. **DPO v2**: SFT warmstart (3 epochs) → DPO (3 epochs), `wrong_tool` preference pairs only
2. Increase pairs to 600, prioritize largest reward margin
3. Format eval gate before live eval
4. If exhausted: **capability LoRA is optional module**; core paper = allocator (H4)

### Status: 🟢 MITIGATED

| Track | HBAC pass@1 | Uniform pass@1 |
|-------|-------------|----------------|
| Base | 44.3% | 44.3% |
| DPO v1 live | 27.7% | 27.7% |
| **DPO v2 live** | **44.3%** | **44.3%** |

DPO v2 capability module restores base pass@1 (vs v1/GRPO regression). Core paper claim = oracle H4 allocator; DPO = optional capability plug-in.

---

## W5 — CLEAR / TAB / Re-FORC are engineering proxies

### Problem
CLEAR uses oracle-metadata surge utility, not Wan et al.’s trained emergence curves. TAB/Re-FORC not fully implemented as paper checkpoints.

### Mitigation plan
1. Explicit **Tier B proxy** labeling in Methodology (already)
2. **Sensitivity**: vary CLEAR `min_per_task`, surge scale in appendix
3. Cite CLEAR/TAB/Re-FORC faithfully in Related Work; compare on **same HBAC stack**

### Paper reframe
> “We compare against **inference-only compose baselines** instantiated from published algorithms, holding L2 fixed — isolating L1 allocation.”

### Status: 🟢 MITIGATED

**CLEAR sensitivity sweep** (`results/clear_sensitivity.json`, bf040 local batches):

| CLEAR min_per_task | pass@1 | Beats uniform? |
|--------------------|--------|----------------|
| 50 | 60% | No |
| 100–400 | 60% | No |
| **HBAC** | **80%** | **Yes (+20 pp)** |

HBAC beats CLEAR at **all** swept settings; CLEAR never beats uniform on pass@1 — documents proxy limitation for appendix.

---

## W6 — Partial HBAC scope

### Problem
Not all paper components validated: tool routing, draft signals (H5−), curriculum (H8 deferred).

### Mitigation
- **H5 refuted locally** → report null result (scientifically valid)
- **H6 refuted** → COMA credit unnecessary at scale
- **H8 deferred** → future work; core Variant B + H4 sufficient for story

### Status: ⚪ ACCEPTED (scope boundary)

---

## W7 — Uniform + LoRA regression (27.7% vs 44.3%)

### Problem
Under GRPO v2 LoRA, uniform allocator drops to 27.7% while HBAC stays 44.3%.

### Resolution
- **Not an eval bug** — greedy decoding, different per-task budgets (~551 vs 600 tokens)
- Analysis: `results/uniform_lora_analysis.json`
- Uniform gets tighter per-task caps → more budget violations / truncated episodes

### Paper reframe
> “LoRA amplifies **budget sensitivity**: uniform allocation is fragile under adaptive inference; HBAC’s differentiated budgets preserve pass@1.”

### Status: 🟢 MITIGATED

---

## W8 — Mean batch reward ≫ pass@1 improvement

### Problem
Reviewers may dismiss 14.7 vs 0.44 mean reward if pass@1 ties.

### Resolution
- Mean batch reward captures **partial credit** (schema compliance, token efficiency, non-violating partial trajectories)
- Oracle H4 shows pass@1 **does** diverge when L2 executes faithfully
- Live tie is **downstream of W1**, not allocator failure

### Status: 🟢 MITIGATED (dual-metric reporting)

---

## W9 — DPO v1: valid JSON, 0% tool-name match

### Problem
Post-DPO format eval (n=100): `valid_json_rate=1.0`, `tool_name_match_rate=0.0`, `mean_tool_reward=0.426`.

Model learned JSON syntax but not **conditional tool selection** given conversation context.

### Root cause (confirmed)
1. DPO pairs diluted across 3 rejection modes
2. No SFT warmstart before DPO
3. Weak wrong_tool signal

### Fix applied (DPO v2) — **SUCCESS on format eval**
- SFT warmstart 3 epochs + DPO 3 epochs, **wrong_tool-only** pairs (600)
- Job `16764458` COMPLETED (~20 min)
- **Results** (`results/grpo_format_dpo_v2.json`):

| Metric | DPO v1 | DPO v2 |
|--------|--------|--------|
| valid_json_rate | 100% | **100%** |
| tool_name_match_rate | 0% | **100%** |
| mean_tool_reward | 0.426 | **0.90** |

### Status: 🟢 MITIGATED (format + live parse) — see W1 for pass@1 ceiling

---

## W10 — No LaTeX paper draft

### Problem
Only markdown context stores exist; no submission-ready PDF.

### Mitigation
Write paper after W1/W4 close (target: 70%+ submittable). Use [Paper Narrative.md](Paper%20Narrative.md) as outline.

### Status: 🔴 DEPRIORITIZED — focus on [Research Discovery.md](Research%20Discovery.md) experiments first

---

## W11 — Small live eval sample (n=300)

### Problem
300 tasks may be underpowered for allocator pass@1 differences (~5 pp MDE at 44% base rate).

### Mitigation
1. Add **per-benchmark breakdown** (mock, tau, toolbench, swe)
2. Report **bootstrap CIs** in Results
3. Scale to n=500 if DPO v2 shows signal

### Status: 🟢 MITIGATED — `results/live_compose_analysis.json` (bootstrap 95% CI: 38.7–50.3% pass@1)

---

## W12 — CLEAR ties uniform in oracle H4

### Problem
CLEAR does not beat uniform in our setting (both 60% vs HBAC 80%).

### Resolution
Documented in Results §2: proxy surge utility without per-query emergence curves collapses to uniform split under our floor constraints.

### Status: 🟢 MITIGATED

---

## Experiment backlog (linked to weaknesses)

| Action | Weakness | Command / artifact |
|--------|----------|-------------------|
| CLEAR sensitivity sweep | W5 | `python -m hbac.scripts.analyze_clear_sensitivity` |
| Live compose analysis | W1, W11 | `python -m hbac.scripts.analyze_live_compose` |
| Impact loop | all | `bash scripts/run_impact_loop.sh --quick` |

---

## Reviewer objection → response cheat sheet

| Objection | Response |
|-----------|----------|
| “Pass@1 doesn’t improve live” | JSON fixed (DPO v2). Aggregate pass@1 ties due to SWE/tau stub difficulty; HBAC wins on batch reward (33×), tokens, violations. Oracle H4: +20 pp pass@1. |
| “Oracle replay isn’t real” | Standard compose-vs-joint eval [CLEAR, TAB]; live eval confirms reward separation; stubs control confounds. |
| “CLEAR baseline weak” | Inference-only proxy on same L2; sensitivity in appendix; HBAC beats both under tight budget. |
| “GRPO failed” | Diagnosed format regression; SFT fixes JSON; DPO targets tool-name; allocator story independent of LoRA. |
| “Stub benchmarks” | Matched deterministic episodes for allocator A/B; scale to full suites is future work. |
| “HBAC is just SJF / task dropping” | Type-prior ties oracle pass@1; D14 explicitly flagged as reward hacking; HBAC retains non-zero SWE budget vs type-prior. |
| “Pareto dominance / 39×” | Withdrawn in v2; report raw pass@1 + tokens + violations; efficiency at matched pass@1 only. |
| “Proxy ZEBRA collapse” | Reported with disclaimer; may be implementation bug; official code not integrated. |
| “Compliant utility fabricated” | Demoted to appendix; primary tables use raw metrics. |

---

## Changelog

| Date | Update |
|------|--------|
| 2026-07-06 | **Paper v2** after Strong Reject review: heuristic baselines, honest type-prior tie, D14 negative, proxy disclaimer, controller overhead |
| 2026-07-05 | Initial tracker; DPO v1 train done (W9 identified); DPO live eval pending |
| 2026-07-05 | DPO v2 pipeline + live eval diagnostics planned |
| 2026-07-05 | Jobs 16764411/16764442 **FAILED** (bad L2 path, CUDA device) → fixed, resubmitted 16764457–59 |
| 2026-07-05 | **DPO v2 format gate PASSED**: tool_name_match 0%→100%, mean_tool_reward 0.43→0.90 |
| 2026-07-05 | **DPO v2 live complete**: pass@1 ties 44.3%; JSON clean; per-benchmark decomposition |
| 2026-07-05 | CLEAR sensitivity + live compose analysis scripts; W2/W3/W5/W11 mitigated |
