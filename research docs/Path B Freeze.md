# Path B Freeze — HBAC NeurIPS submission framing

**Frozen:** 2026-07-21 (updated 2026-07-23)  
**Status:** Paper claims locked under Path B. SWE salvage **closed** (still 0% after seed/prompt/fuzzy). Coder-Next live slice **retry** pinned to A100-80GB.

---

## Primary claims (must appear in abstract + §5)

1. **Holdout live (clean DPO):** `hbac_d18` vs `type_prior` = **+0.45 pp** (26.80% vs 26.35%), McNemar p≈0.0039, paired CI [0.20, 0.75] pp, n=2000, floor=400.  
   Artifact: `results/rivanna/compose_live_v3_holdout_floor400_n2000.json`

2. **Oracle:** HBAC/D18 **80%** ties type-prior; +20 pp vs uniform/Tier-A; guardrail hurts (74.7%).

3. **Capability precondition:** Qwen3-Coder-Next-FP8 (vLLM) uniform LCB **77.5–87.5%**; SWE **0%** under local harness → live allocator study scoped to **LCB + τ + toolbench**.

4. **D18:** starvation penalty removes need for inference `fairness_reserve_alloc` on oracle.

5. **Ethics:** type_prior starve_rate=1.0 / ~0% LCB share; hbac_d18 retains ~15.4% LCB share (`budget_share_starvation.json`).

## Secondary / demoted

- Contaminated DPO v2 live **+1.3 pp** (report with contamination table; not primary).
- SWE ≥5% plan gate: **FAIL / CLOSED** — local gold, prompt fix, and fuzzy salvage all 0%; future Docker SWE-bench Verified only.

## Venue

- NeurIPS **Main / General** (conditions characterization) **or** **Evaluations & Datasets** (harness + paired protocol + ethics).
- Do **not** submit as Negative Results unless reframed as impossibility/conditions science with higher originality bar.

## Do not re-run

- McNemar already significant on holdout — do not scale n for significance.
- Do **not** chase more local SWE harness variants.
- Capability pilots already prove LCB as hard solvable class.

## Stretch (non-blocking)

1. ~~Fuzzy SWE salvage~~ — **done, FAIL** (job 17192331: LCB 82.5%, SWE 0%).
2. Coder-Next live slice (hbac_d18 vs type_prior, LCB+τ+toolbench) — **COMPLETED** on A100-80GB (job 17233651): hbac_d18 pass@1 = **56.9%** vs type_prior = **20.0%** on n=160; paired McNemar p≈3.47e-18 (directional supported). See `results/paired_allocator_analysis_coder_next_slice.json`.
