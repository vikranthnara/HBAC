"""ZEBRA additive water-filling allocator — Tier-A faithful to arXiv:2605.20485 [A5].

Saturating-exponential phase utility f_i(x) = a_i(1 - exp(-b_i x)).
Per-phase allocation x_i(λ) = max(0, (1/b_i) ln(a_i(b_i + λ)/λ)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hbac.baselines.clear import _normalize_to_budget
from hbac.training.batch_curriculum import BatchTask


def _phase_curve(task: BatchTask) -> tuple[float, float]:
    """Estimate (a_i, b_i) from oracle metadata."""
    a = min(0.95, 0.40 + 0.45 / max(task.difficulty, 0.5))
    b = max(0.001, 2.0 / max(task.oracle_tokens, 100))
    return a, b


def waterfill_allocation(task: BatchTask, lam: float) -> float:
    a, b = _phase_curve(task)
    if lam <= 0:
        return float(task.oracle_tokens)
    # Starve if log-marginal at zero below λ
    if (a * b) / max(1e-9, 1.0 - a) < lam:
        return 0.0
    ratio = a * (b + lam) / lam
    if ratio <= 1.0:
        return 0.0
    return max(0.0, (1.0 / b) * math.log(ratio))


@dataclass
class ZEBRAOfficialAllocator:
    """Bisection on Lagrange multiplier λ until Σ x_i = B."""

    min_per_task: int = 40
    max_iters: int = 40

    def allocate(self, tasks: list[BatchTask], global_budget: int) -> dict[str, int]:
        if not tasks:
            return {}
        n = len(tasks)
        if global_budget < self.min_per_task * n:
            base = max(1, global_budget // n)
            return {t.task_id: base for t in tasks}

        lo, hi = 0.0, 10.0
        best: dict[str, int] | None = None
        for _ in range(self.max_iters):
            mid = (lo + hi) / 2.0
            raw = {t.task_id: int(waterfill_allocation(t, mid)) for t in tasks}
            alloc = _normalize_to_budget(raw, global_budget, 0)
            demand = sum(alloc.values())
            if demand > global_budget:
                lo = mid
            else:
                hi = mid
                best = alloc

        if best is None:
            base = max(1, global_budget // n)
            return {t.task_id: base for t in tasks}
        return _normalize_to_budget(best, global_budget, self.min_per_task)


__all__ = ["ZEBRAOfficialAllocator", "waterfill_allocation"]
