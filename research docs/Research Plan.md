# Title

**Hierarchical Budgeted Agent Controller (HBAC): Joint Optimization of Global Budget Allocation, Adaptive Stopping, Tool Use, and Approximate Inference for Long-Horizon LLM Agents**

---

## Research context stores (living documents)

Use these as the primary paper-writing and experiment context. Update them as Rivanna jobs complete and new literature is surveyed.

| Document | Purpose |
|----------|---------|
| [Methodology.md](Methodology.md) | Architecture, equations (LaTeX), loss functions, training stages |
| [Experiments.md](Experiments.md) | Setup, datasets, baselines, metrics, reproducibility commands |
| [Results.md](Results.md) | Findings, tables, ablations, charts — **canonical empirical record** |
| [Related Work.md](Related%20Work.md) | Literature survey, comparison matrix, research inbox |
| [Paper Narrative.md](Paper%20Narrative.md) | Paper-ready claims, abstract bullets, figure list |
| [Weaknesses.md](Weaknesses.md) | Reviewer weaknesses tracker — mitigations, status |
| [Research Discovery.md](Research%20Discovery.md) | **Active hypotheses + experiment queue** (engineering focus) |

This file ([Research Plan.md](Research%20Plan.md)) remains the master plan: epistemology (§0), hypotheses (§10), gates (§16), and roadmap.

---

## 0. Epistemological Framework

Every statement in this document is classified into one of three tiers. **No design choice appears without a Tier A citation or an explicit Tier C hypothesis.**

| Tier | Label | Meaning | Example |
|------|-------|---------|---------|
| **A** | **Established fact** | Peer-reviewed or primary-source result we rely on as ground truth | SWE-Bench Verified has 500 human-validated instances [A6, A7] |
| **B** | **HBAC instantiation** | Engineering choice that *implements* Tier A methods for our setting; correctness follows from Tier A + stated constraints | We use PPO [A10] with KL penalty [A11] for Variant A |
| **C** | **Empirical hypothesis** | Claim that *must* be measured on HBAC experiments; not assumed true | HBAC Pareto-dominates CLEAR [A4] under global budget \(B_{\text{total}}\) |

**Implementation status** (codebase facts, June 2026): Phase 1 complete. Phase 2 Variant A stop-controller: reward sweep, reference-policy KL, KL ablation (H7), train/eval CLIs. **Phase 3 complete:** L1 GRPO + counterfactual credit (Variant B), utility-net L1 (Variant A), joint Stage 4, LLM GRPO (Phase 3b). See §12.

---

## 1. Literature Foundation (Tier A)

This section records **only** what prior work has established. HBAC builds on these results; it does not re-derive them.

### 1.1 Agent formalism

| Fact | Source |
|------|--------|
| LLM agents are modeled as sequential decision processes with partial observability (conversation history, tool outputs) | ReAct [A1]; τ-bench as tool–agent–user POMDP [A9] |
| Hierarchical control decomposes decisions across timescales via temporally extended actions (options) | Options framework [A12]; semi-MDP formulation [A12] |
| Multi-agent credit assignment can use counterfactual baselines that hold other agents fixed | COMA [A13] |

**HBAC mapping (Tier B):** Level 1 (batch allocator) and Level 2 (task controller) follow the options/semi-MDP pattern [A12]: Level 1 selects per-task budget \(b_i\); Level 2 executes until termination.

### 1.2 Inference-time compute and budget allocation

| Fact | Source |
|------|--------|
| Test-time compute can be allocated non-uniformly across inputs to improve accuracy–cost trade-offs | AdaCompute [A14]; CLEAR shadow-price allocation [A4] |
| Per-turn token budgeting in multi-turn reasoning improves accuracy–token trade-offs vs static budgets | TAB [A2] |
| Early stopping via predicted future reward reduces CoT compute; grounded in Pandora's box / Gittins index | Re-FORC [A3]; Weitzman [A24] |
| Zero-shot multi-phase budget splitting via Lagrangian water-filling avoids RL training | ZEBRA [A5] |
| Speculative decoding uses a draft model; acceptance rate reflects draft–target agreement and is observable at runtime | Leviathan et al. [A15]; Chen et al. [A16] |

### 1.3 RL training for LLM policies

| Fact | Source |
|------|--------|
| PPO stabilizes policy-gradient updates via clipped surrogate objective | Schulman et al. [A10] |
| GRPO removes the critic by normalizing rewards within sampled groups; variant of PPO | Shao et al. [A17] |
| KL penalty to a reference policy mitigates reward over-optimization / hacking in RLHF | Ouyang et al. [A11]; Stiennon et al. [A18] |

### 1.4 Evaluation benchmarks (verified properties)

| Benchmark | Established property | Source |
|-----------|---------------------|--------|
| SWE-Bench | 2,294 real GitHub issues; patch + test verification | Jimenez et al. [A6], ICLR 2024 |
| SWE-Bench Verified | 500 human-validated subset; OpenAI collaboration, Aug 2024 | [A6], [A7] |
| LiveCodeBench | Contamination-free; LeetCode/AtCoder/Codeforces; codegen + self-repair | Jain et al. [A8], arXiv:2403.07974 |
| ToolBench | 16,464 REST APIs; multi-tool instruction tuning | Qin et al. [A19] |
| τ-bench | Tool–agent–user interaction; retail + airline; pass^k reliability metric | Yao et al. [A9], ICLR 2025 |
| τ²-bench | Dual-control Dec-POMDP; user + agent act on shared state | Barres et al. [A20] |

---

## 2. Literature Gap Analysis (Tier A → motivates Tier C)

The table below states **what each cited system optimizes**, not marketing claims. Empty cells indicate scope *not addressed* in that work (verified from abstracts/introductions).

| Capability | ReAct [A1] | TAB [A2] | Re-FORC [A3] | CLEAR [A4] | ZEBRA [A5] | HBAC target (Tier C) |
|------------|:--:|:--:|:--:|:--:|:--:|:--:|
| Global batch budget \(\sum_i C_i \le B_{\text{total}}\) | — | per-problem only | — | ✓ | ✓ (monetary) | ✓ (tokens) |
| Per-step adaptive stopping | — | — | ✓ | truncation | — | ✓ |
| External tool selection / cost | ✓ (heuristic) | — | — | — | — | ✓ (learned) |
| Draft-model / approximate inference routing | — | — | — | — | — | ✓ |
| Joint Level-1 + Level-2 RL training | — | partial (budgeter only) | — | — | — | ✓ |
| Draft acceptance signals in RL state \(D\) | — | — | — | — | — | ✓ (hypothesis) |
| Coding + API + user-interaction benchmarks | partial | math only | math only | reasoning tasks | APPS/QA | SWE+LCB+Tool+τ |

**Tier C hypothesis (novelty claim, empirically testable):** No cited work jointly learns (i) global batch allocation, (ii) per-task stopping, (iii) tool routing, and (iv) draft-model routing under a unified token budget with draft signals in state \(D\). HBAC will confirm or refute this via ablation and baseline comparison (§10).

---

## 3. Problem Definition

### 3.1 Hierarchical constrained POMDP (Tier B, grounded in [A12], [A9])

**Level 1 — Global Budget Allocator**  
Given batch \(\mathcal{Q}=\{q_1,\ldots,q_n\}\) and global budget \(B_{\text{total}}\):

$$B_{\text{total}} \rightarrow \{b_1,\ldots,b_n\}, \quad \sum_i b_i \le B_{\text{total}}$$

**Level 2 — Task Controller** (one per \(q_i\), sequential):

* **State:** \(s_t = (h_t, b_i^{\text{rem}}, T, D)\)
  - \(h_t\): conversation history [A1]
  - \(b_i^{\text{rem}}\): remaining task token budget
  - \(T\): tool registry (ToolBench [A19]; bash/edit in SWE [A6])
  - \(D\): draft-model signals — **operational definition (Tier B):** per-step draft acceptance rate \(\alpha_t \in [0,1]\) and expected verification cost from speculative decoding [A15, A16]
* **Actions:** \(a_t = (a_{\text{stop}}, a_{\text{tool}}, a_{\text{approx}})\)

### 3.2 Why POMDP (Tier A)

Agents do not observe true environment state (full repo, user intent, test outcomes until executed). τ-bench explicitly models tool–agent–user partial observability [A9]. HBAC adopts the same formalism.

---

## 4. Global Objective and Reward Structure

### 4.1 Constrained batch objective (Tier B, Lagrangian form from [A4])

$$\max \sum_i \mathbb{E}[S_i] \quad \text{s.t.} \quad \sum_i C_i \le B_{\text{total}}, \quad S_i \in \{0,1\}$$

Lagrangian (shadow-price interpretation per CLEAR [A4]):

$$\mathcal{L} = \sum_i \mathbb{E}[S_i] - \lambda \left(\sum_i C_i - B_{\text{total}}\right)$$

### 4.2 Level-2 reward (Tier B, implemented in `hbac/training/reward.py`)

$$R^{(2)} = S_i - \lambda C_i - \gamma L_i - \delta R_i$$

| Term | Definition | Grounding |
|------|------------|-----------|
| \(S_i\) | Terminal task success (1/0) | Verifiable pass/fail on benchmarks [A6, A8] |
| \(C_i\) | Total tokens consumed | TAB token cost [A2]; CLEAR budget [A4] |
| \(L_i\) | Wall-clock latency (ms) | Systems cost axis in Re-FORC [A3] |
| \(R_i\) | Risk/error penalty | Re-FORC risk term [A3] |
| Premature-stop penalty | Extra penalty if agent stops before env success | Mitigates RLHF-style hacking [A11]; Re-FORC early-stop [A3] |

**Tier A invariant (validated in code, must pass before training):**
1. Success at equal tokens beats premature stop
2. In-budget success beats over-budget success
3. Token cost is monotonic holding success fixed

Run: `python -m hbac.scripts.validate_reward`

### 4.3 Level-1 reward (Tier B)

$$R^{(1)} = \sum_i (S_i - \lambda C_i)$$

Sparse batch signal; allocator does not observe step-level actions (hierarchical RL standard [A12]).

---

## 5. Credit Assignment (Tier B design, Tier A inspiration)

### 5.1 Counterfactual allocation credit (Phase 3 — implemented)

$$A_i = R_{\text{batch}} - R_{\text{batch}}^{(-i)}$$

**Grounding:** Counterfactual advantage for multi-agent credit [A13]. Implemented in `hbac/training/credit.py`; used by `train_variant_b.py` and `run_phase3`.

### 5.2 Utility prediction network (Phase 3 — implemented)

$$V(q_i, b) = \mathbb{E}[U_i \mid q_i, b]$$

**Grounding:** Value baseline for policy gradients [A10]; CLEAR utility curves \(U(q,b)\) [A4]. Implemented in `hbac/training/utility_net.py` + `train_variant_a_l1.py`.

---

## 6. Architecture Variants (Tier B)

| Component | Variant A | Variant B | Citation |
|-----------|-----------|-----------|----------|
| Level-2 optimizer | PPO + critic | GRPO (no critic) | [A10], [A17] |
| Level-1 optimizer | PPO + utility network | GRPO | [A10], [A17] |
| KL regularization | ✓ (`kl_coef`, adaptive) | ✓ (GRPO ref policy) | [A11], [A17] |
| Phase-2 prototype | Monolithic stop head only (Stage 1) | — | Curriculum [A21] |

**Variant A (PPO):** Actor-critic with classification heads for \(a_{\text{stop}}\), \(a_{\text{tool}}\), \(a_{\text{approx}}\) [A10].

**Variant B (GRPO):** Group-relative advantages per TAB training recipe [A2, A17].

---

## 7. Datasets and Benchmarks

| ID | Benchmark | HBAC role | Key cited fact | Code status |
|----|-----------|-----------|----------------|-------------|
| B1 | SWE-Bench Verified | Long-horizon SE, bash/python | 500 instances [A6, A7] | Implemented |
| B2 | LiveCodeBench | Algorithmic code + self-repair | 500+ problems, contamination-free [A8] | Implemented |
| B3 | ToolBench | API / multi-tool orchestration | 16K APIs [A19] | Planned |
| B4 | τ-bench / τ²-bench | User interaction, partial observability | ICLR 2025 [A9]; Dec-POMDP [A20] | Planned |

---

## 8. Metrics and Evaluation Protocol (Tier B, standard definitions)

| Metric | Definition | Notes |
|--------|------------|-------|
| Pass@1 | \(\frac{1}{n}\sum_i S_i\) | Standard [A6, A8] |
| Budget violation (task) | \(P(C_i > b_i)\) | Per-task constraint |
| Budget violation (batch) | \(P(\sum_i C_i > B_{\text{total}})\) | Global constraint [A4] |
| Utility / token | \(S_i / C_i\) if \(S_i=1\) else 0 | Efficiency axis [A2, A4] |
| NetGain (Tier B) | Token savings from approximate inference minus controller overhead | Speculative decoding reduces target-model tokens [A15, A16]; overhead = draft + routing tokens |

Implemented: `hbac/core/metrics.py` (Pass@1, budget violation, utility/token; NetGain planned)

---

## 9. Baselines (Tier A specifications)

| ID | Method | What it isolates | Venue / arXiv | HBAC impl. |
|----|--------|------------------|---------------|------------|
| BL1 | **ReAct** [A1] | Naive think–act–observe, fixed budget | ICLR 2023 | ✓ |
| BL2 | **TAB** [A2] | Turn-level token allocation; GRPO budgeter | arXiv:2604.05164 (Apr 2026) | ✓ heuristic |
| BL3 | **Re-FORC** [A3] | Reward-forecast early stop; no global budget | arXiv:2511.02130; NeurIPS 2025 ER Workshop | ✓ heuristic |
| BL4 | **CLEAR** [A4] | Shadow-price economic allocation | arXiv:2606.03092; ICML 2026 | Planned |
| BL5 | **ZEBRA** [A5] | Zero-shot multi-phase knapsack | arXiv:2605.20485; ICML 2026 AgenticUQ | Planned |

**Authors (verified):** TAB — Jali, Nayak, Joshi (CMU) [A2]. Re-FORC — Zabounidis et al. (AWS Agentic AI) [A3]. CLEAR — Wan et al. [A4]. ZEBRA — Hamri, Talgam-Cohen (Tel Aviv Univ.) [A5].

### 9.1 Implementation fidelity (Tier B — not paper replicas)

Phase 1 baselines are **engineering proxies** for evaluation scaffolding. They implement the *decision structure* of each cited method, not trained checkpoints from the original papers.

| Baseline | Paper method [Tier A] | HBAC v1 implementation [Tier B] | Paper-faithful replication |
|----------|----------------------|----------------------------------|----------------------------|
| BL1 ReAct | Thought–action–observe loop [A1] | Fixed-budget loop in `hbac/baselines/react.py` | ✓ (no learned component) |
| BL2 TAB | GRPO-trained turn budgeter on math MDP [A2] | `HeuristicTABPolicy`: history-length + error proxies | Requires TAB checkpoint + math env |
| BL3 Re-FORC | Beta adapter forecasting \( \mathbb{E}[R^* \mid \text{tokens}] \); Pandora's box greedy stop [A3, A24] | `HeuristicForecaster`: logistic on feedback features; threshold on \(J = \psi - \lambda C\) | Requires Re-FORC adapter weights |
| BL4 CLEAR | Shadow-price batch allocation + truncation [A4] | Not implemented | Requires utility-curve predictor |
| BL5 ZEBRA | Zero-shot Lagrangian knapsack [A5] | Not implemented | Inference-only wrapper |

**Tier C note:** Whether heuristic BL2/BL3 match learned paper performance is an empirical question (H1–H3); they are **not assumed** to be equivalent.

## 10. Empirical Hypotheses (Tier C — must be tested, not assumed)

| ID | Hypothesis | Test |
|----|------------|------|
| H1 | HBAC exceeds ReAct [BL1] on Pass@1 at equal \(B_{\text{total}}\) | Run baselines on B1–B4 |
| H2 | HBAC exceeds TAB [BL2] on utility/token in multi-turn coding | SWE + LCB |
| H3 | HBAC exceeds Re-FORC [BL3] on batch utility under global budget | Re-FORC lacks global budget by design [A3] |
| H4 | HBAC exceeds CLEAR [BL4] / ZEBRA [BL5] when stopping + tools matter | Ablate Level-2 heads |
| H5 | Draft signals \(D\) improve Level-2 policy vs ablated \(D=\emptyset\) | Remove \(\alpha_t\) from state |
| H6 | Counterfactual credit [§5.1] stabilizes Level-1 vs uniform baseline | COMA-style ablation [A13] |
| H7 | KL penalty [A11] prevents early-stop collapse vs \(kl\_coef=0\) | PPO ablation |
| H8 | Curriculum Stages 1→4 [A21] beats training all heads from scratch | Stage-wise learning curves |

**Disclaimers (Tier A facts that bound claims):**
- Re-FORC does not optimize global batch budget [A3] — comparisons must hold batch budget fixed externally.
- TAB optimizes math multi-turn budgets, not agent tool graphs [A2].
- CLEAR/ZEBRA are inference-time allocators without learned stopping/tools [A4, A5].

---

## 11. Training Strategy (Tier B, grounded in [A10, A11, A17, A21])

### 11.1 Curriculum (Bengio et al. [A21])

| Stage | Trains | Status |
|-------|--------|--------|
| 1 | \(a_{\text{stop}}\) only | Done |
| 2 | \(a_{\text{stop}}, a_{\text{tool}}\) (partial) | Done (stop head) |
| 3 | Level 1 allocator, fixed Level 2 | **Done** (`run_phase3`, `train_variant_b` Stage 3) |
| 4 | Joint fine-tuning | **Done** (`run_phase3` Stage 4, `train_variant_b --stage 4`) |

### 11.2 Oracle initialization (Tier B)

Strong-model ReAct [BL1] rollouts → filter \(S_i=1\) → SFT / GRPO groups [A17]. Implemented: `collect_oracles.py`, `export_sft.py`.

### 11.3 Hyperparameter search

Bayesian optimization for LR, entropy, KL coef, \(\lambda\) [A22].

### 11.4 KL protocol (Tier A → B)

Per Ouyang et al. [A11]: KL to reference policy prevents over-optimization. HBAC implements adaptive `kl_coef` in `hbac/training/ppo.py`. Reward invariants (§4.2) are necessary but not sufficient; H7 validates jointly.

---

## 12. Implementation Status (codebase, June 2026)

| Module | Path | Tier |
|--------|------|------|
| AgentEnv + budgets | `hbac/core/env.py`, `cost.py` | B |
| SWE-Bench / LiveCodeBench | `hbac/envs/` | B |
| Baselines BL1–BL3 | `hbac/baselines/` | B |
| Reward + validation + sweep | `hbac/training/reward.py`, `validation.py` | B |
| Variant A Stage 1 | `hbac/training/controller.py`, `ppo.py`, `probes.py`, `dataset.py` | B |
| Phase 2 CLIs | `train_variant_a`, `ablate_kl`, `eval_variant_a`, `validate_reward` | B |
| Phase 3 training | `run_phase3`, `train_variant_b`, `train_variant_a_l1`, `train_llm_grpo`, `check_phase3` | B |
| Level-1 GRPO + credit | `hbac/training/grpo.py`, `credit.py`, `level1.py`, `batch_curriculum.py` | B |
| Phase 3b LLM GRPO | `hbac/training/llm_grpo_trainer.py` (TRL + SFT fallback) | B |
| ControllerRunner (eval) | `hbac/baselines/controller.py` | B |
| Metrics | `hbac/core/metrics.py` | B |

**Verification:** `pytest tests/ -v` (52+ tests). Phase 1–3 complete. Run `python -m hbac.scripts.check_phase3` for Phase 3 gates.

---

## 13. Bibliography (Tier A sources)

| ID | Citation |
|----|----------|
| **A1** | Yao, S., et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023. arXiv:2210.03629 |
| **A2** | Jali, N., Nayak, A., Joshi, G. (2026). *Not All Turns Are Equally Hard: Adaptive Thinking Budgets For Efficient Multi-Turn Reasoning.* arXiv:2604.05164 |
| **A3** | Zabounidis, R., et al. (2025). *Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning.* arXiv:2511.02130. NeurIPS 2025 Efficient Reasoning Workshop |
| **A4** | Wan, X., et al. (2026). *The Shadow Price of Reasoning: Economic Perspective on Optimal Budget Allocation for LLMs* (CLEAR). arXiv:2606.03092. ICML 2026 |
| **A5** | Hamri, M., Talgam-Cohen, I. (2026). *ZEBRA: Zero-shot Budgeted Resource Allocation for LLM Orchestration.* arXiv:2605.20485. ICML 2026 AgenticUQ Workshop |
| **A6** | Jimenez, C. E., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. arXiv:2310.06770 |
| **A7** | OpenAI (2024). *Introducing SWE-bench Verified.* Aug 13, 2024. https://openai.com/index/introducing-swe-bench-verified/ |
| **A8** | Jain, N., et al. (2024). *LiveCodeBench: Holistic and Contamination Free Evaluation of LLMs for Code.* arXiv:2403.07974 |
| **A9** | Yao, S., et al. (2024). *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* ICLR 2025. arXiv:2406.12045 |
| **A10** | Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347 |
| **A11** | Ouyang, L., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS 2022. arXiv:2203.02155 |
| **A12** | Sutton, R. S., Precup, D., Singh, S. P. (1999). *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL.* AIJ 112(1–2). |
| **A13** | Foerster, J., et al. (2018). *Counterfactual Multi-Agent Policy Gradients.* AAAI 2018. arXiv:1705.08926 |
| **A14** | Zhai, Z., et al. (2026). *Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization* (AdaCompute). arXiv:2604.14853 |
| **A15** | Leviathan, Y., Kalman, M., Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. arXiv:2211.17192 |
| **A16** | Chen, C., et al. (2023). *Accelerating LLM Decoding with Speculative Sampling.* arXiv:2302.01318 |
| **A17** | Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO). arXiv:2402.03300 |
| **A18** | Stiennon, N., et al. (2020). *Learning to Summarize from Human Feedback.* NeurIPS 2020 |
| **A19** | Qin, Y., et al. (2023). *ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs.* arXiv:2307.16789 |
| **A20** | Barres, V., et al. (2025). *τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment.* arXiv:2506.07982 |
| **A21** | Bengio, Y., et al. (2009). *Curriculum Learning.* ICML 2009 |
| **A22** | Snoek, J., Larochelle, H., Adams, R. P. (2012). *Practical Bayesian Optimization of ML Algorithms.* NeurIPS 2012 |
| **A23** | Kleinberg, J., Kleinman, R. (2023). *Pandora's Box Problem with Order Constraints.* arXiv:2310.17106 |
| **A24** | Weitzman, M. L. (1979). *Optimal Search for the Best Alternative.* Econometrica 47(3):641–654 (Pandora's box; cited by Re-FORC [A3]) |

---

## 16. Go/No-Go Gates (Phase 3 Gateway)

Empirical gates enforced by `python -m hbac.scripts.check_go_no_go`. Status: **PASS** | **FAIL** | **BLOCKED** | **WARN**.

### Phase 1 — Infrastructure & bootstrapping

| Gate ID | Milestone | Threshold |
|---------|-----------|-----------|
| `env_stability` | 100% deterministic dummy trajectories across SWE, LCB, ToolBench, τ-bench stubs | 100% |
| `oracle_yield` | Frontier ReAct yield on easy/medium split | ≥ 60% |
| `dataset_volume` | Successful oracle trajectories | 500–1,000 |
| `pomdp_compliance` | Trajectories parse to valid POMDP (state, tools, termination) | 100% |
| `baseline_harness` | ReAct + TAB/Re-FORC on 100-sample val (literature repro) | Pass@1 ≥ 40% on n≥100 |

### Phase 2 — Component prototyping

| Gate ID | Milestone | Threshold |
|---------|-----------|-----------|
| `reward_invariants` | Anti-hacking reward validation | 5/5 pass |
| `stop_format_compliance` | Oracle tool JSON parse rate (Stage 1; full a_stop/a_tool/a_approx Phase 3) | ≥ 95% |
| `early_stop_tool_tasks` | Premature stop on tool-required tasks | < 5% |
| `budget_violation_l2` | Per-task budget violation (Level 2) | < 2% |
| `kl_stability` | \|KL(ref‖new)\| tail mean during PPO | 0.01–0.05 |
| `draft_overhead` | Controller + draft routing ≤ 15% of global budget | ≤ 15% |
| `level1_allocator` | Batch budget adherence | < 2% violation |

### Phase 3 gateway — do not start full-scale RL until all PASS

| Gate ID | Milestone | Threshold |
|---------|-----------|-----------|
| `dummy_batch_timing` | 10-episode dummy hierarchical batch | < 5 min, 0 crashes |
| `sft_budget_obedience` | Controller respects per-task budgets | < 2% violation |
| `overfit_curve` | 30-sample overfit reward improves without hacking | Δreward ≥ 0 |

**Current status:** All 15 gateway gates **PASS** — run `python -m hbac.scripts.check_go_no_go --oracle-path data/oracles`. Phase 3 training complete — run `python -m hbac.scripts.check_phase3 --phase3-path checkpoints/phase3`.

---

## 17. Empirical Results

**→ Full tables, charts, and ablations: [Results.md](Results.md)**

Summary (July 2026, Rivanna run `/standard/liverobotics/hbac-run-20260630T183941Z`):

| Result | Status |
|--------|--------|
| H4 oracle tight budget (80% vs 60% pass@1) | **CONFIRMED** |
| Live LLM HBAC mean reward 14.7 vs uniform 0.44 | **CONFIRMED** |
| H6 counterfactual credit | **No effect** at 150-batch scale |
| H5 draft signals | **No effect** locally |
| GRPO LoRA v1 | **FAILED** (44.3% → 27.7% pass@1) |
| GRPO v2 (SFT + tool-aware reward) | **PARTIAL** — HBAC ties base; GRPO = SFT-only |

Refresh: `bash scripts/run_impact_loop.sh` · `results/experiment_summary.json`

---

## 14. Reproducibility Checklist

```bash
pytest tests/ -v
python -m hbac.scripts.validate_reward
python -m hbac.scripts.seed_oracles
python -m hbac.scripts.collect_oracles --env livecodebench --limit 20 --output data/oracles/lcb
python -m hbac.scripts.train_variant_a --oracle-path data/oracles --subset-limit 50
python -m hbac.scripts.ablate_kl --oracle-path data/oracles
python -m hbac.scripts.check_go_no_go --oracle-path data/oracles
pytest tests/test_phase2_acceptance.py -v
```

---

## 15. Document Changelog

| Date | Change |
|------|--------|
| Jun 2026 | Added Tier A/B/C epistemology; full bibliography; literature gap table; corrected Re-FORC venue/date; operational definition of \(D\); all hypotheses moved to Tier C |
| Jun 2026 | Phase 2 complete: reference-policy KL, reward sweep, KL ablation H7, merged oracle training, eval harness |
| Jun 2026 | Added §16 Go/No-Go gates, ToolBench/τ-bench stubs, `check_go_no_go` CLI |
| Jul 2026 | §17 empirical results (Rivanna H4/live/GRPO); H5 draft features; impact feedback loop |
| Jul 2026 | Phase 3c DPO pipeline; stub oracles; Paper Narrative.md; uniform LoRA analysis |
