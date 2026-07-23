# Research Discovery Log

**Status: Path B FROZEN. SWE salvage CLOSED (0%). CN live slice retry on A100-80GB submitted.**

*Last updated: July 23, 2026*

See **`research docs/Path B Freeze.md`** for locked primary claims.

---

## 5/5 scorecard (Path B)

| Criterion | Status |
|-----------|--------|
| McNemar holdout +0.45 pp | ✅ PRIMARY |
| D18 / theory / ethics / holdout DPO | ✅ |
| Hard-task capability via LCB | ✅ Coder-Next 77.5–87.5% |
| SWE ≥5% | ❌ **CLOSED** (fuzzy 17192331 still 0%) |

---

## Jobs

| Job | Result |
|-----|--------|
| 17192331 fuzzy SWE | COMPLETED — LCB 82.5%, SWE 0% |
| 17192332 CN live | FAILED — A100-40GB, vLLM init fail |
| **NEW** CN live 80GB (job 17233651) | COMPLETED — hbac_d18 56.9% vs type_prior 20.0% on n=160; paired McNemar p≈3.47e-18. See `results/paired_allocator_analysis_coder_next_slice.json`. |

---

## Changelog

| Date | Update |
|------|--------|
| 2026-07-23 | Close SWE salvage; retry CN live with `-C a100_80gb` |
| 2026-07-21 | Path B freeze; fuzzy+CN submit; prompt fix LCB 87.5% |
| 2026-07-20 | vLLM LCB 82.5% |
| 2026-07-15 | Holdout live +0.45 pp |
