# HBAC: Hierarchical Budgeted Agent Control Under Token Scarcity

**Paper v2 draft** — revised after adversarial review (Strong Reject → scope correction)  
**Status:** Camera-ready PDF available — [`Paper v2.pdf`](Paper%20v2.pdf) (LaTeX: [`../paper/main.tex`](../paper/main.tex))  
**Artifact manifest:** `results/canonical_artifacts.json` (v2/v3 block)

---

## Abstract

When multiple LLM agent tasks share a **global token budget**, allocators must decide per-task caps and when to stop reasoning. We present **Hierarchical Budgeted Agent Control (HBAC)**, a two-stage system: a **Level-1 schema allocator** (GRPO-trained) sets batch budgets, and a **Level-2 stop head** (PPO-trained) halts frozen ReAct rollouts. We evaluate on a **V3 real pool** (LiveCodeBench + SWE Lite oracles, τ-bench, ToolBench) with **Tier-A official** CLEAR/ZEBRA/Re-FORC baselines, plus a legacy **stub micro-harness** for controlled ablations.

On **oracle replay** at 40% global budget, HBAC achieves **80% pass@1** vs **60%** for uniform and CLEAR (+20 pp, n=300–500 tasks). A **type-prior heuristic** **ties HBAC at 80%** on oracle but **starves** LCB/SWE budget; **hbac_fair** (fairness reserve) trades oracle pass@1 (74.4%) for non-zero hard-task allocation. On **live** Qwen2.5-7B + DPO v2 LoRA at **floor=400**, **hbac_fair beats type-prior by +1.3 pp** (27.65% vs 26.35%, n=2000; CIs overlap) — the gap is **floor-invariant** across 300–600 and driven by **LiveCodeBench** (3.25% vs 0%), not oracle allocation. Legacy stub eval (n=300) shows pass@1 **ties at generous floors** (+27.6 pp vs uniform only under tight floors). We report **negative results** openly: D14 ROI-skip is reward hacking; `hard_min_frac` sweep does not close the oracle gap to type-prior; SWE pass@1 remains 0% live. Controller overhead is **~19 µs per allocation**. We release code, locked artifacts, heuristic baselines, and a reproducibility checklist.

---

## 1. Introduction

### 1.1 Motivation

Batch agent evaluation—running SWE-bench, ToolBench, τ-bench, and related suites under a shared GPU-hour or token quota—is increasingly common. **Uniform** allocation ignores heterogeneous difficulty; **per-query economic allocators** (CLEAR, ZEBRA) optimize single-task emergence curves and may not respect batch structure; **per-turn methods** (TAB, Re-FORC) do not allocate across tasks.

### 1.2 What this paper is (and is not)

| In scope | Out of scope (v2/v3) |
|----------|------------------------|
| V3 real pool (LCB + SWE Lite oracles, n=2000 live) | Full production SWE-bench / LCB agent stacks |
| Oracle replay on curriculum + real_eval batches | Claiming Pareto dominance or 39× SOTA |
| Live eval (Qwen2.5-7B + DPO v2 LoRA; stub n=300 + V3 n=2000) | Joint end-to-end training of L1+L2+LLM |
| Tier-A official CLEAR/ZEBRA/Re-FORC + heuristics | Statistical dominance over type-prior (CIs overlap) |

### 1.3 Contributions (revised)

1. **HBAC architecture** — Decoupled training: GRPO Level-1 batch allocator + PPO Level-2 stop head over **frozen** ReAct rollouts (contextual bandit + stop classifier, §3.1).
2. **Oracle H4** — +20 pp pass@1 vs uniform/CLEAR (80% vs 60%); **type-prior ties** hbac_joint at 80% on oracle; **hbac_fair** 74.4% with softer hard-task allocation (§5.1).
3. **V3 live @ floor=400 (n=2000)** — **hbac_fair beats type-prior +1.3 pp**; floor dose-response shows +1.0–1.7 pp at all floors 300–600 (§5.4–5.5).
4. **Per-benchmark decomposition** — Aggregate win is **LCB-only** (type-prior zeros LCB live); τ/toolbench tied; SWE 0% all allocators (§5.4).
5. **Live dual-metric (stub n=300)** — pass@1 ties @ floor=600; +27.6 pp @ floor=400 vs uniform; efficiency + violation-free allocation (§5.2).
6. **Expanded baselines** — Tier-A official CLEAR/ZEBRA/Re-FORC; SJF, type-prior, proxies; proxy disclaimer (§4.4, §5.3).
7. **Honest negative results** — D14 reward hacking; `hard_min_frac` ablation cannot close oracle gap; fairness risks (§6–7).
8. **Reproducibility** — Locked artifacts, controller overhead, DPO mixture (Appendix).

### 1.4 Paper roadmap

| Section | Content |
|---------|---------|
| §2 | Related work |
| §3 | Method (revised formalism) |
| §4 | Experimental setup |
| §5 | Results (raw metrics first) |
| §6 | Ablations & negative results |
| §7 | Limitations & fairness |
| §8 | Conclusion |
| App. | Compliant utility, DPO mixture, reproduction |

---

## 2. Related Work

### 2.1 Batch-level allocation

| Method | Mechanism | Our comparison |
|--------|-----------|----------------|
| **CLEAR** | Shadow-price bisection on surge utility | **Tier-B proxy**; oracle ties uniform (60%); live harmful (violations, negative reward) |
| **ZEBRA** | Water-filling on utility curves | **Tier-B proxy**; oracle 60%; live 0% pass@1 @ floor=400 (possible proxy bug—see §7) |
| **Uniform** | Equal split | Strong baseline; collapses under tight floors |

### 2.2 Per-turn / per-chain control

| Method | Scope | Our comparison |
|--------|-------|----------------|
| **TAB** | Per-turn budgets (math) | **BatchTABProxyAllocator** — difficulty-weighted caps |
| **Re-FORC** | Gittins early stopping | **BatchReFORCProxyAllocator** — marginal-utility threshold |

We do **not** claim to beat official TAB/Re-FORC implementations; proxies test whether batch-level analogues close the gap.

### 2.3 Agent training

ReAct provides frozen L2 action policy; GRPO/DPO LoRA (Phase 3c) fixes tool-JSON for live eval. Allocator optimality (oracle) is **decoupled** from generation competence (DPO).

---

## 3. Method

### 3.1 Problem formulation (revised — not a joint hierarchical POMDP)

We clarify terminology per reviewer feedback:

- **Level 1** is a **contextual bandit** over discrete allocation *schemas* \(\pi^{(1)}(s \mid \mathcal{Q}, B_{\text{total}})\), where context features encode batch composition (benchmark mix, oracle lengths, difficulty). GRPO updates schema logits; counterfactual credit (\(\beta=0.2\)) is a **variance-reduction heuristic**, not an unbiased gradient estimator—we report ablations without formal unbiasedness proofs.
- **Level 2** is a **binary stop classifier** over states from **frozen** ReAct rollouts. L2 does not co-train the LLM policy; episodes follow a fixed Markov chain conditioned on per-task budget.

Joint hierarchical POMDP language applies only if L1, L2, and the LLM policy are trained jointly—which we explicitly do **not** do in v2.

Given batch \(\mathcal{Q}\) and budget \(B_{\text{total}}\):

$$\{b_1,\ldots,b_n\} = \pi^{(1)}(\mathcal{Q}, B_{\text{total}}), \quad \sum_i b_i \le B_{\text{total}}$$

Per-task rollout stops when \(\pi^{(2)}_{\text{stop}}(s_t)=1\), budget exhausted, or success.

### 3.2 Rewards (training)

**Level-2 (terminal):** \(R^{(2)}_i = S_i - \lambda C_i - \gamma L_i - \delta R_i - \eta \cdot \mathbb{1}[\text{premature stop}]\)

**Level-1 (batch, Variant B GRPO):**

$$R^{(1)} = \text{pass\_rate} - \lambda_v \cdot \text{violations} + \beta \cdot \text{Var}_b(\text{budget}_b)$$

Optional parse penalty (D16): falsified for live improvement.

### 3.3 Primary evaluation metrics (v2)

All main tables report **raw**:

1. **pass@1** — task success rate  
2. **mean tokens used** — per task or per batch  
3. **batch violation rate** — budget overruns  
4. **mean batch reward** — \(R^{(1)}\) from training objective  

**Compliant utility** (Appendix A only): \(U = R \cdot (1 - \text{violation\_rate}) - 0.5 \cdot \text{parse\_failures}\). The 0.5 weight is an **engineering penalty** for deployment hygiene, not a theoretically derived utility; we ablate raw metrics in all primary comparisons.

### 3.4 Training pipeline

1. Stage 1–2: L2 stop head (PPO, KL-regularized)  
2. Stage 3 (Variant B): L1 GRPO over allocation schemas  
3. Phase 3c: DPO v2 capability LoRA (see Appendix B for data mixture)

### 3.5 Controller overhead

**Artifact:** `results/controller_overhead.json`

| Metric | Value |
|--------|-------|
| L1 allocation latency | **~19 µs** per batch (500 repeats, numpy policy) |
| Proxy control tokens | **~1** per batch |
| Typical tokens saved vs uniform (live) | **~94** per task |
| Control-to-savings ratio | **~1:94** |

Dominant cost is LLM generation, not HBAC controller FLOPs.

---

## 4. Experimental Setup

### 4.1 Benchmarks — stub micro-harness

Live and oracle eval use a **heterogeneous stub mix**: ToolBench, τ-bench, mock, SWE-bench **stubs** (deterministic trajectories from `collect_stub_oracles.py`). This is **not** full SWE-bench Lite or LiveCodeBench.

| Benchmark | Stub pass@1 ceiling (all allocators) |
|-----------|-------------------------------------|
| toolbench | ~100% |
| τ-bench | ~33% |
| SWE-bench stub | **0%** |

Claims about "heterogeneous agent benchmarks" must be read as **controlled stubs**, not production agent stacks.

### 4.2 Models & training

| Component | Setting |
|-----------|---------|
| L2 | Frozen stop controller (Variant B tight bf040) |
| L1 | GRPO schema policy (`local_tight_bf040`, seed47) |
| Live LLM | Qwen2.5-7B-Instruct + DPO v2 LoRA (`20260705T014948Z_capability_v2`) |
| Budget fraction | 40% global (primary) |

### 4.3 Eval protocols

| Protocol | n | Purpose |
|----------|---|---------|
| Oracle replay (tight bf040) | 300 tasks, 30 batches | Primary pass@1 + heuristic matrix |
| Live compose | 300 tasks, 50 batches | Qwen rollouts, floor ablations |
| Bootstrap 95% CI | 2000 resamples | pass@1 uncertainty |

### 4.4 Baselines (v2 matrix)

| Tier | Allocators |
|------|------------|
| **Standard** | Uniform, HBAC |
| **Tier-B proxies** | CLEAR, ZEBRA, BatchTAB, BatchRe-FORC |
| **Heuristics (new)** | SJF, type-prior, difficulty-inverse |

**Proxy disclaimer:** CLEAR, ZEBRA, TAB, and Re-FORC implementations follow published mechanisms but are **not** official author codebases. ZEBRA's 0% pass@1 @ floor=400 may reflect proxy water-filling bugs under extreme floors—we report it but do not treat it as definitive SOTA collapse.

### 4.5 DPO v2 and test-set leakage

DPO v2 uses **wrong_tool** rejection pairs from oracle trajectories (`data/oracles`), 600 pairs, SFT warmstart—see Appendix B. Pairs are built from **training oracles**, not live eval task IDs; we document the mixture in `results/capability_report.json` but cannot rule out benchmark-family overlap without external audit.

---

## 5. Results

### 5.1 Oracle replay — tight heterogeneous batches (PRIMARY)

**Artifacts:** `results/rivanna/compose_tight_bf040_seed47.json`, `results/v2_baseline_matrix_oracle_tight.json`

| Allocator | pass@1 | mean batch reward | violations | tokens/task |
|-----------|--------|-------------------|------------|-------------|
| **HBAC** | **80.0%** | 1.02 | 0% | — |
| Uniform | 60.0% | 0.60 | 0% | — |
| CLEAR (proxy) | 60.0% | 0.15 | 9.1% | — |
| ZEBRA (proxy) | 60.0% | 0.90 | 0% | 47.3 |
| SJF | 40.0% | 0.44 | 0% | 48.0 |
| **Type-prior** | **80.0%** | **1.35** | 0% | 37.0 |
| Difficulty-inverse | 60.0% | 0.17 | 0% | 53.1 |
| TAB proxy | 60.0% | 0.16 | 0% | 54.4 |
| Re-FORC proxy | 60.0% | 0.15 | 0% | 55.6 |

**Interpretation:**

- HBAC **+20 pp** vs uniform/CLEAR on pass@1 under scarcity—**primary supported claim**.
- **Type-prior ties HBAC** on pass@1 and achieves **higher batch reward** by explicitly zeroing SWE budget. HBAC learns a **softer** version of the same structural insight without hand-coded benchmark lists—a useful but **not dominant** result.
- SJF underperforms (40%)—shortest-job-first alone is insufficient on this mix.
- TAB/Re-FORC proxies do not beat uniform on pass@1 in this batch-level mapping.

HBAC allocation variance: **1456** vs uniform **0**—differentiated budgets matter.

#### Oracle vs live: why type-prior ties on oracle but loses live (n=2000)

**V3 real pool oracle** (`results/rivanna/v3_real_oracle_matrix.json`, n=500): hbac_joint **80%**, type-prior **80%**, hbac_fair **74.4%**, uniform/Tier-A **60%**. On oracle replay, success is determined by **deterministic trajectories**—type-prior's zero LCB/SWE budget does not hurt pass@1 because hard tasks were already unsolvable in the oracle index.

**Live generation** breaks this equivalence: Qwen2.5-7B + DPO v2 can solve **some** LiveCodeBench problems when allocated budget. Type-prior **zeros LCB allocation live** → **0% LCB pass@1**; hbac_fair reserves **hard_min_frac=15%** → **3.25% LCB**. The aggregate +1.3 pp win is therefore a **generation-regime effect**, not an oracle allocation gap. A `hard_min_frac` sweep (0.10–0.25) on the real_eval oracle pool holds hbac_fair at **74.7%** and **never beats** type-prior (80%)—confirming the live win is not recoverable by tuning fairness alone on oracle (§6.5).

### 5.2 Live evaluation — dual regime (raw metrics)

**Model:** Qwen2.5-7B + DPO v2 | **n=300** | Bootstrap 95% CI for HBAC pass@1 ≈ **38.7–50.3%**

#### Regime A: Generous floor (floor=600)

**Artifact:** `results/rivanna/compose_live_bf040_seed47_dpo_v2.json`

| Allocator | pass@1 | Mean reward | Tokens/task | Violations |
|-----------|--------|-------------|-------------|------------|
| HBAC | 44.3% | **14.70** | **504** | **0%** |
| Uniform | 44.3% | 0.44 | 598 | 0% |
| CLEAR | 44.3% | −0.27 | 677 | **14.3%** |

- **pass@1 ties** — stub ceilings dominate (toolbench 100%, τ 33%, SWE 0%).
- HBAC leads on **batch reward** (~33× vs uniform) and **token efficiency** (~94 fewer tokens/task)—not on pass@1.
- We **withdraw** "Pareto dominance" and "39× reward per token" as headline claims; report as **efficiency separation at matched pass@1**.

#### Regime B: Tight floor (floor=400)

**Artifact:** `results/rivanna/compose_live_bf040_floor400_dpo_v2.json`

| Allocator | pass@1 | Mean reward | Tokens |
|-----------|--------|-------------|--------|
| **HBAC** | **44.3%** | **6.79** | 400 |
| Uniform | **16.7%** | 0.17 | 400 |
| CLEAR | 44.3% | −0.27 | 480 |

**+27.6 pp** HBAC vs uniform. Gap emerges when uniform cannot fund multi-step tool chains; **not** observed @ floor=600.

#### Floor dose-response

| Floor | HBAC pass@1 | Uniform | Gap (pp) |
|-------|-------------|---------|----------|
| 600 | 44.3% | 44.3% | 0 |
| 500 | 44.3% | 27.7% | +16.6 |
| **400** | **44.3%** | **16.7%** | **+27.6** |
| 300 | 27.7% | 0% | +27.7 |

### 5.3 Proxy baselines @ floor=400

**Artifact:** `results/rivanna/compose_live_bf040_floor400_all_baselines_dpo_v2.json`

| Allocator | pass@1 |
|-----------|--------|
| HBAC | 44.3% |
| Uniform | 16.7% |
| CLEAR | 44.3% |
| ZEBRA (proxy) | 0.0% |

ZEBRA collapse is **suspect** until validated with official code—we note it as a proxy artifact.

### 5.4 Heuristic baselines (live — n=2000, floor=400)

**Artifact:** `results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json`  
**Model:** Qwen2.5-7B + DPO v2 | **n=2000** | `fairness_reserve` (D17) | Bootstrap 95% CI

| Allocator | pass@1 | 95% CI |
|-----------|--------|--------|
| **HBAC joint** | **27.95%** | 26.1–29.9% |
| **HBAC fair** | **27.65%** | 25.8–29.6% |
| Type-prior | 26.35% | 24.5–28.2% |
| ZEBRA (proxy) | 22.15% | 20.4–24.0% |
| Re-FORC official | 16.35% | 14.8–18.0% |
| CLEAR official | 11.60% | 10.3–13.0% |
| Uniform / CLEAR / TAB / Re-FORC proxy | ~11.3% | ~10.0–12.7% |
| ZEBRA official | 9.2% | 8.0–10.5% |
| SJF | 4.3% | 3.4–5.2% |

**Interpretation:**

- **HBAC fair beats type-prior** by **+1.3 pp** on pass@1 (`hbac_fair_beats_type_prior=true`). CIs overlap — a modest separation, not a dominant win.
- HBAC joint is marginally above fair (+0.3 pp); fairness reserve does not materially hurt pass@1 at this scale.
- Uniform and compose proxies collapse to ~11% — the tight-floor regime separates learned allocators from naive baselines.
- Type-prior remains the strongest heuristic competitor; HBAC fair's value is learning allocation **without** hand-coded benchmark taxonomy while matching/beating type-prior.
- Oracle matrix (§5.1) still shows type-prior **ties** HBAC at 80%; live n=2000 is the first scale where fair reserve edges type-prior.

**Pilot (n=300):** `compose_live_v3_pilot_floor400_dpo_v2.json` — same direction (+1.0 pp), overlapping CIs; n=2000 confirms at full scale.

#### Per-benchmark pass@1 (n=2000, floor=400)

| Benchmark | n | HBAC joint | HBAC fair | Type-prior | Uniform |
|-----------|---|------------|-----------|------------|---------|
| toolbench | 400 | 100% | 100% | 100% | 50% |
| τ-bench | 400 | 31.75% | 31.75% | 31.75% | 0% |
| livecodebench | 800 | **4.0%** | **3.25%** | **0%** | 3.25% |
| swe_bench | 400 | 0% | 0% | 0% | 0% |

- **Source of +1.3 pp:** LCB only — type-prior starves hard tasks under live generation; hbac_fair keeps residual budget.
- τ-bench: all learned allocators tie; uniform collapses (cannot fund multi-step chains).
- SWE: 0% ceiling for all allocators on this model/checkpoint mix.

### 5.5 HBAC fair vs type-prior — floor dose-response (V3 pool, n=300)

**Artifact:** `results/rivanna/fair_floor_sweep_shards/` | **Analysis:** `results/fair_floor_sweep_analysis.json`  
**Job:** Rivanna `16832736` (floors 300–600, `hbac_fair` + `type_prior` only)

| Floor | HBAC fair | 95% CI | Type-prior | 95% CI | Gap (pp) | CIs overlap? |
|-------|-----------|--------|------------|--------|----------|--------------|
| 300 | 26.7% | 21.7–32.0% | 25.3% | 20.7–30.7% | +1.3 | Yes |
| 350 | 26.3% | 21.7–31.7% | 25.3% | 20.7–30.7% | +1.0 | Yes |
| **400** | **26.3%** | 21.7–31.7% | **25.3%** | 20.7–30.7% | **+1.0** | Yes |
| 450 | 27.0% | 22.0–32.3% | 25.3% | 20.7–30.7% | +1.7 | Yes |
| 500 | 26.3% | 21.7–31.7% | 25.3% | 20.7–30.7% | +1.0 | Yes |
| 600 | 26.3% | 21.7–31.7% | 25.3% | 20.7–30.7% | +1.0 | Yes |

**Interpretation:**

- **HBAC fair beats type-prior at all 6 floors** (mean gap **+1.2 pp**; range +1.0 to +1.7 pp).
- Separation is **floor-invariant** on the V3 real pool — unlike HBAC-joint vs uniform (§5.2), which only separates under tight floors.
- Type-prior pass@1 is **flat ~25.3%** across floors; hbac_fair is similarly flat ~26.3–27.0%.
- Per-floor CIs overlap; the n=2000 @ floor=400 result (+1.3 pp) provides scale confirmation but not statistical dominance.
- **Narrative:** Oracle type-prior **ties** hbac_joint at 80% by starving SWE; live hbac_fair **edges** type-prior by allocating some budget to hard tasks while still biasing tools — a softer, learned version of the same structural insight.

---

## 6. Ablations & Negative Results

### 6.1 Hypothesis tests

| ID | Verdict | Notes |
|----|---------|-------|
| H4 | ✅ | +20 pp oracle vs uniform/CLEAR |
| H5 | ❌ | Draft signals neutral |
| H6 | ❌ | COMA credit neutral |
| H7 | ✅ | KL prevents stop collapse |
| D6 | ❌ | ControllerRunner = ReAct |
| D16 | ❌ | Parse-penalty L1 oracle-neutral |

### 6.2 D14 ROI-skip — reward hacking (negative)

**Artifact:** `results/rivanna/compose_live_bf040_floor300_roi_skip_dpo_v2.json`

D14 zeros SWE budget below floor=350 and redistributes to ToolBench. This is **shortest-job-first task dropping**, not learned optimal allocation. We report +27.7 pp @ floor=300 as an **upper bound on heuristic cheating**, not a contribution. **Do not deploy.**

### 6.3 D12 scarcity boost (inference, not dropping)

**D12 refined:** +17% batch reward vs baseline @ floor=400 with **0 parse failures** (`compose_live_bf040_floor400_scarcity_refined_dpo_v2.json`). Shifts budget SWE→tool with `swe_min_reserve=0.5`—partial reallocation, not zeroing.

### 6.4 Type-prior ties HBAC (oracle)

The strongest honest finding: a **20-line heuristic** matches HBAC pass@1 on oracle. HBAC's value proposition shifts to:
- Learning without benchmark taxonomy labels  
- Softer fairness (non-zero hard-task budget vs type-prior starvation)  
- Live edges (+1–1.7 pp) where generation makes LCB partially solvable

### 6.5 `hard_min_frac` oracle ablation (D17 sensitivity)

**Artifact:** `results/hard_min_frac_oracle_sweep.json` | **Pool:** `data/oracles/real_eval/latest`, n=300 tasks

| hard_min_frac | hbac_fair pass@1 | Gap vs type-prior |
|---------------|------------------|-------------------|
| 0.10 | 74.7% | −5.3 pp |
| 0.15 (default) | 74.7% | −5.3 pp |
| 0.20 | 74.7% | −5.3 pp |
| 0.25 | 74.7% | −5.3 pp |

Type-prior: **80.0%** (fixed). **Verdict:** `NEVER_BEATS_TYPE_PRIOR` — fairness reserve trades oracle pass@1 for allocation diversity; tuning `hard_min_frac` does not close the gap. The n=2000 live win over type-prior is **not** explained by oracle allocation superiority.

---

## 7. Limitations & Fairness

1. **Benchmark coverage** — V3 adds real LCB/SWE Lite oracles; SWE live pass@1 still **0%**; τ/toolbench partially stubbed. Full production agent stacks remain future work.
2. **Live scale & significance** — n=2000 @ floor=400 completed; hbac_fair vs type-prior gap (+1.3 pp) has **overlapping CIs** — directionally consistent across 6 floors but not statistically dominant.
3. **Oracle–live disconnect** — Type-prior ties on oracle (80%) but loses live LCB; hbac_fair cannot beat type-prior on oracle at any `hard_min_frac` tested.
4. **Tier-A vs Tier-B** — V3 integrates official CLEAR/ZEBRA/Re-FORC; legacy proxy results retained for stub ablations.
5. **Type-prior parity (oracle)** — Learned allocator does not dominate simple heuristics on oracle pass@1; live edge is modest and LCB-specific.
6. **Fairness / starvation** — Type-prior zeros hard tasks live; hbac_fair mitigates but SWE remains unfunded. Production systems need **per-class minimum guarantees**.
7. **DPO mixture** — Documented but not externally audited for eval leakage.
8. **Joint training** — L2 frozen ReAct limits "hierarchical RL" claims.

---

## 8. Conclusion

HBAC provides a **reproducible harness** for batch-level token allocation and early stopping under global scarcity. Oracle evaluation shows **+20 pp pass@1** over uniform/CLEAR; a **type-prior heuristic ties** hbac_joint at 80% on oracle—an important honesty check. **V3 live evaluation (n=2000)** shows **hbac_fair edges type-prior by +1.3 pp** at floor=400, confirmed across a six-floor dose-response (+1.0–1.7 pp), with the gap localized to **LiveCodeBench** where type-prior starves budget under live generation. We explicitly reject reward-hacking (D14), report that `hard_min_frac` tuning cannot close the oracle gap, and scope claims away from statistical dominance or SWE SOTA. Code, Tier-A baselines, heuristic allocators, overhead measurements, and locked artifacts support a publication-ready narrative focused on **honest allocation tradeoffs** rather than headline pass@1 wins.

---

## Appendix A: Compliant utility (secondary metric)

$$U = R \cdot (1 - \text{violation\_rate}) - 0.5 \cdot \text{parse\_failures}$$

Rationale: violations zero utility (hard constraint); parse failures penalized at 0.5 as a deployment hygiene knob. **Primary tables use raw pass@1, tokens, violations.** Compliant utility ranking (`results/compliant_utility_matrix.json`) is supplementary.

---

## Appendix B: DPO v2 data mixture

| Field | Value |
|-------|-------|
| Oracle root | `data/oracles` |
| Pairs | 600 wrong_tool rejections |
| SFT warmstart | 3 epochs |
| DPO epochs | 3 |
| Tool-name match (format eval) | 100% |
| Checkpoint | `20260705T014948Z_capability_v2` |

Source: `results/capability_report.json`, `hbac/training/llm_dpo_trainer.py`

---

## Appendix C: Reproduction commands

```bash
# V2 oracle baseline matrix (tight bf040)
python -m hbac.scripts.eval_compose_v2 \
  --batches-path checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/batches.jsonl \
  --l1-checkpoint checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz \
  --l2-checkpoint checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/frozen_l2_controller.npz \
  --output results/v2_baseline_matrix_oracle_tight.json

# Controller overhead
python -m hbac.scripts.analyze_controller_overhead \
  --batches-path checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/batches.jsonl \
  --l1-checkpoint checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz \
  --output results/controller_overhead.json

# Live eval with heuristics (GPU/API)
python -m hbac.scripts.eval_compose_live \
  --batches-path ... --l1-checkpoint ... --l2-checkpoint ... \
  --live-min-per-task 400 --lora-path checkpoints/.../capability_v2

# hard_min_frac oracle ablation (CPU)
python -m hbac.scripts.analyze_hard_min_frac_oracle \
  --batches-path checkpoints/eval_real/batches.jsonl \
  --oracle-path data/oracles/real_eval/latest \
  --output results/hard_min_frac_oracle_sweep.json
```

---

## Appendix D: Artifact index (v2)

| Claim | File |
|-------|------|
| Oracle H4 (+20 pp) | `results/rivanna/compose_tight_bf040_seed47.json` |
| V2 heuristic matrix | `results/v2_baseline_matrix_oracle_tight.json` |
| Controller overhead | `results/controller_overhead.json` |
| Live efficiency @ floor=600 | `results/rivanna/compose_live_bf040_seed47_dpo_v2.json` |
| Live +27.6 pp @ floor=400 | `results/rivanna/compose_live_bf040_floor400_dpo_v2.json` |
| D14 negative (ROI skip) | `results/rivanna/compose_live_bf040_floor300_roi_skip_dpo_v2.json` |
| D12 refined | `results/rivanna/compose_live_bf040_floor400_scarcity_refined_dpo_v2.json` |
| V3 live n=2000 @ floor=400 | `results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json` |
| V3 fair floor sweep | `results/fair_floor_sweep_analysis.json` |
| V3 oracle matrix (real pool) | `results/rivanna/v3_real_oracle_matrix.json` |
| hard_min_frac oracle ablation | `results/hard_min_frac_oracle_sweep.json` |

*Manifest: `results/canonical_artifacts.json` (v2/v3 block)*
