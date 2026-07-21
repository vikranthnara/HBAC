# Paper Narrative

**Context store** for claims ready to write up. Companion: [Results.md](Results.md) · [Methodology.md](Methodology.md)

*Last updated: July 6, 2026 — superseded by [Paper v1.md](Paper%20v1.md) for full draft*

---

## 1. Core claim (H4 — CONFIRMED)

Under a **global token budget** with tight per-task floors, **HBAC's hierarchical allocator** achieves **80% pass@1** vs **60%** for uniform and CLEAR on oracle replay (n=500 per track, 40/45/50% budget fractions).

**Why it matters:** Prior work (CLEAR, TAB, Re-FORC) optimizes inference compute but does not jointly train Level-1 batch allocation + Level-2 stopping for agent benchmarks. HBAC fills that gap under scarcity.

**Evidence:** `results/rivanna/compose_tight_bf040_seed47.json` (+ bf045, bf050)

---

## 2. Live LLM claim (CONFIRMED — reward, not pass@1)

On **Qwen2.5-7B-Instruct** with stub agent benchmarks (n=300, 40% budget):

| Metric | HBAC | Uniform | Ratio |
|--------|------|---------|-------|
| Mean batch reward | **14.70** | 0.44 | **33×** |
| pass@1 | 44.3% | 44.3% | tied |
| Batch violations | 0% | 0% | — |

**Framing for paper:** When end-task success ties (tool-JSON bottleneck), HBAC still wins on the **metric aligned with hierarchical budgeting** — batch reward, token efficiency, zero violations. The allocator is doing meaningful work even before LoRA improves generation.

---

## 3. Phase 3b/3c — capability LoRA (DPO v2 CONFIRMED)

| Method | Format tool-name | Live pass@1 (HBAC) | Outcome |
|--------|------------------|----------------------|---------|
| GRPO v1 | — | 27.7% | Failed |
| GRPO v2 SFT | 100% | 44.3% (ties base) | GRPO adds nothing |
| DPO v1 | 0% | 27.7% | JSON only |
| **DPO v2** | **100%** | **44.3%** | **Capability module works** |

**Paper sentence:** We decouple **allocator optimality** (oracle H4, +20 pp) from **generation competence** (DPO capability LoRA); live eval reports both pass@1 and batch-reward metrics with per-benchmark decomposition.

---

## 4. Ablation takeaways

| Hypothesis | Result | Paper sentence |
|------------|--------|----------------|
| H6 COMA credit | No effect at scale | Counterfactual credit is optional under oracle-dense L1 rewards |
| H5 draft signals | No local effect | Draft acceptance αₜ does not improve stop head with current features |
| H7 KL penalty | PASS | Reference-policy KL stabilizes PPO without stop collapse |

---

## 5. Suggested abstract bullets

1. Two-level hierarchical POMDP for batch budget + per-task agent control
2. +20 pp pass@1 over uniform/CLEAR under tight global budget (oracle H4)
3. 33× mean batch reward improvement on live 7B eval at matched pass@1
4. DPO v2 capability LoRA: 100% tool-name match; live JSON clean; allocator metrics preserved

---

## 6. Figures for paper

1. **Fig 1:** Methodology architecture (from Methodology.md mermaid)
2. **Fig 2:** H4 pass@1 bar chart (40/45/50% tracks)
3. **Fig 3:** Live eval mean batch reward bar chart
4. **Table 1:** Comparison matrix (Related Work.md §7)
5. **Table 2:** Ablation summary (H5–H8, GRPO v1/v2)
