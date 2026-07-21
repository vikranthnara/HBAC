# External baseline repositories

Author codebases for CLEAR, ZEBRA, and Re-FORC were **not publicly available** as of July 2026
(CLEAR [Code] link on Xu Wan's homepage points to forthcoming ICML 2026 release).

## Tier-A (in-repo, paper-faithful)

| Paper | Module | Algorithm |
|-------|--------|-----------|
| CLEAR (Wan et al., ICML 2026) | `hbac/baselines/clear_official.py` | Lambert W + bisection shadow price |
| ZEBRA (arXiv:2605.20485) | `hbac/baselines/zebra_official.py` | Saturating-exp water-filling |
| Re-FORC (NeurIPS 2025) | `hbac/baselines/reforc_official.py` | Gittins marginal-ψ allocation |

## Tier-B (legacy proxies)

| Module | Notes |
|--------|-------|
| `hbac/baselines/clear.py` | Oracle-metadata surge proxy |
| `hbac/baselines/zebra.py` | Weighted proportional fill |
| `hbac/baselines/heuristics.py` | SJF, type-prior, TAB/Re-FORC batch proxies |

## When official repos publish

```bash
git submodule add <CLEAR_URL> external/clear
git submodule add <ZEBRA_URL> external/zebra
# Wire thin adapters in hbac/baselines/vendor/
```

Until then, eval scripts report both Tier-A and Tier-B rows for regression checking.
