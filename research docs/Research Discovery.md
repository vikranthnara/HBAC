# Research Discovery Log

**Status: Path H — SWE harness fixed; vLLM re-pilot submitted.**

*Last updated: July 21, 2026*

---

## 5/5 success-criteria scorecard

| Reviewer demand | Status | Evidence |
|-----------------|--------|----------|
| McNemar p<0.05 | ✅ PASS | holdout +0.45 pp, p≈0.004 |
| D18 no inference hack primary | ✅ PASS | oracle starvation 0; live D18≡joint |
| **SWE pass@1 ≥5%** | ⏳ re-pilot | Prior 0% was empty-workspace harness; gold-seed fix landed |
| Theory bounds + β sweep | ✅ PASS | appendix + credit_beta_sweep.json |
| DPO audit (0 overlap) | ⚠️ PARTIAL | LCB=0 on holdout; stub residual FAIL |
| Tier-A only in oracle table | ✅ PASS | paper Table oracle-v3 |
| TAB separate live row | ✅ PASS | 11.30% row |
| Ethics + starvation metrics | ✅ PASS | budget_share_starvation.json + Table |

---

## Capability stack

| Job | Result |
|-----|--------|
| 17142925 vLLM Coder-Next-FP8 | **COMPLETED** — LCB **82.5%**, SWE **0%** (broken harness) |
| **NEW** vLLM + SWE harness fix | See `results/rivanna/capability_2xa100_jobs.json` |

**Harness root cause:** `swe_env_for_task` used `local_mode=True` on an empty tempfile; success=`bool(patch)`. Fixed in `hbac/envs/swe_local.py`: seed pre-patch files from gold unified diff + grade against post-patch contents (micro fallback for `swe-local-1`).

---

## Locked allocator results (do not re-run for significance)

- Holdout primary: **+0.45 pp**, McNemar p≈0.004
- Contaminated DPO secondary: +1.3 pp

---

## Changelog

| Date | Update |
|------|--------|
| 2026-07-21 | Path H: SWE gold-seed harness; re-pilot submit |
| 2026-07-20 | vLLM pilot COMPLETED (LCB 82.5%, SWE 0%) |
| 2026-07-16 | 1×A6000 4-bit FAILED; 2×A100 Path A |
| 2026-07-15 | Holdout live COMPLETE (+0.45 pp) |
