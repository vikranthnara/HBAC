# Path B Freeze — HBAC NeurIPS submission framing

**Frozen:** 2026-07-21 (updated 2026-07-23)  
**Status:** Paper claims locked under Path B. SWE salvage **closed**. Coder-Next live slice **integrated** into `paper/main.tex` as stretch. Packaging locked via `results/canonical_artifacts.json`.

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

## Stretch (reported, not primary)

Coder-Next live slice (job **17233651**, A100-80GB): `hbac_d18` **56.9%** vs `type_prior` **20.0%** (+36.9 pp, n=160, McNemar p≈3.47e-18; LCB-localized).  
Artifact: `results/paired_allocator_analysis_coder_next_slice.json`  
Paper: §Live evaluation stretch subsection + Table `tab:cn-slice`.

## Venue

- Prefer **NeurIPS Main / General** (conditions characterization + paired protocol).
- Acceptable alternate: **Evaluations & Datasets** (open harness + contamination audit + ethics metrics).
- Do **not** submit as Negative Results unless reframed as impossibility/conditions science with higher originality bar.

## Submission packaging checklist

- [x] Holdout primary +0.45 pp in abstract / contributions / live section
- [x] CN stretch subsection + artifact index row
- [x] Capability scoped to LCB; SWE closed / future Docker
- [x] Venue note in conclusion
- [x] `results/canonical_artifacts.json` updated (Path B; five_of_five_ready via LCB path)
- [x] Compile PDF + page count (`cd paper && make pdf`) → **9 pages** (`paper/main.pdf`)
- [ ] NeurIPS paper checklist PDF (author-side)
- [ ] Broader impact / ethics paragraph review (already in draft)

## Do not re-run

- McNemar already significant on holdout — do not scale n for significance.
- Do **not** chase more local SWE harness variants.
- Capability pilots already prove LCB as hard solvable class.
- Full CN n=2000 matrix — optional only if reviewers demand capable-model primary.

## Reproduce

- CPU: `scripts/reproduce_v3.sh`
- GPU: `scripts/rivanna/submit_holdout_live.sh`, `submit_capability_2xa100_vllm.sh`, `submit_coder_next_live_80gb.sh`
- Manifest: `results/canonical_artifacts.json`
