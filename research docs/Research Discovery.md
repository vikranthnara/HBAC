# Research Discovery Log

**Status: Path B FROZEN. Paper packaging in progress (CN stretch wired into `paper/main.tex`). SWE CLOSED. No more GPU science jobs.**

*Last updated: July 23, 2026*

See **`research docs/Path B Freeze.md`** for locked primary claims + packaging checklist.

---

## 5/5 scorecard (Path B)

| Criterion | Status |
|-----------|--------|
| McNemar holdout +0.45 pp | ✅ PRIMARY |
| D18 / theory / ethics / holdout DPO | ✅ |
| Hard-task capability via LCB | ✅ Coder-Next 77.5–87.5% |
| CN stretch live slice | ✅ +36.9 pp (n=160) in paper as stretch |
| SWE ≥5% | ❌ **CLOSED** (fuzzy 17192331 still 0%) |
| Artifact manifest | ✅ `results/canonical_artifacts.json` |

---

## Jobs

| Job | Result |
|-----|--------|
| 17192331 fuzzy SWE | COMPLETED — LCB 82.5%, SWE 0% |
| 17192332 CN live | FAILED — A100-40GB, vLLM init fail |
| 17233651 CN live 80GB | COMPLETED — hbac_d18 56.9% vs type_prior 20.0% (n=160); p≈3.47e-18 |

---

## Next (non-GPU)

1. Compile `paper/` PDF + page check
2. Author NeurIPS checklist
3. Final ethics/broader-impact pass

---

## Changelog

| Date | Update |
|------|--------|
| 2026-07-23 | Wire CN slice into paper; lock canonical_artifacts Path B |
| 2026-07-23 | Close SWE salvage; CN live SUCCESS on a100_80gb |
| 2026-07-21 | Path B freeze; fuzzy+CN submit; prompt fix LCB 87.5% |
| 2026-07-20 | vLLM LCB 82.5% |
| 2026-07-15 | Holdout live +0.45 pp |
