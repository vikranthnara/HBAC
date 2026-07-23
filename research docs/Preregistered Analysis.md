# HBAC V3 — Preregistered Analysis Plan

**Registered:** 2026-07-09 (before new GPU live runs)  
**Scope:** V3 live allocation study (floor=400 primary; floor sweep secondary)

---

## Scope note (Path B, 2026-07-21)

When SWE pass@1 remains 0% under the local harness, **LiveCodeBench** is the preregistered hard-but-solvable class for capability precondition~(i).
Primary live claims stay on LCB+τ+toolbench; SWE is reported as a capability probe / future Docker work, not a primary allocator endpoint.
See `research docs/Path B Freeze.md`.

---

## Primary endpoints

| Endpoint | Definition | Test |
|----------|------------|------|
| **P1** | Paired per-task success: `hbac_d18` vs `type_prior` | Two-sided McNemar; Holm-Bonferroni over {P1, P2} |
| **P2** | Paired per-task success: `hbac_joint` vs `type_prior` | Same |
| **S1** | Per-benchmark pass@1 (LCB, τ, toolbench, SWE) | Descriptive + stratified McNemar where paired vectors exist |
| **S2** | Mean tokens / task | Descriptive |
| **S3** | `starvation_rate` (hard benchmarks below fair floor) | Compare allocators; D18 ladder |

## Success criteria

- **Claim "directionally higher pass@1"** only if McNemar **p < 0.05** (Bonferroni-adjusted) on P1 or P2.
- **Claim "beats type-prior"** is **retracted** unless P1 passes at α=0.05 after correction.
- Report **95% bootstrap CI on paired difference** in pass@1 (paired resample of tasks).

## Futility / stop rules

| Stage | Rule |
|-------|------|
| n=2000 (existing) | Run McNemar; if p > 0.10 and \|Δ\| < 2 pp, do **not** scale GPU for significance alone |
| n=5000 | Run only if power analysis projects ≥80% power at observed discordant-pair rate |
| n=10000 | **Do not run** if McNemar p > 0.10 at n=5000 → publish **negative result** |

## Power analysis protocol

Script: `python -m hbac.scripts.power_analysis_paired`

- Input: paired success vectors or marginal counts from prior run
- Output: estimated discordant pairs (b, c), required n for 80% power, recommendation {scale | stop}

## Allocator naming (v3)

| Key | Definition |
|-----|------------|
| `hbac_d18` | L1 from D18 training (`starvation_penalty`); **no** `fairness_reserve_alloc` at inference |
| `hbac_joint` | Standard L1; no inference post-process |
| `hbac_guardrail` | L1 + `fairness_reserve_alloc` (deprecated ablation only) |
| `type_prior` | `TypePriorAllocator` heuristic |

## Contamination

DPO audit (`results/dpo_contamination_audit.json`) must show **0 exact task-ID overlap** before live claims cite DPO-trained models.

## Reproducibility

- Locked artifacts: `results/canonical_artifacts.json`
- Paired analysis: `results/paired_allocator_analysis.json`
- Power: `results/power_analysis_paired.json`
