"""Re-FORC Gittins-style batch allocator — Tier-A proxy for Zabounidis et al. [A3].

Batch-level mapping: allocate budget proportional to estimated continuation value
ψ(t) = E[R | x, t more tokens]. Uses oracle_tokens and difficulty as ψ proxy;
abandon when marginal ψ < λ (cost-per-token).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hbac.baselines.clear import _normalize_to_budget
from hbac.training.batch_curriculum import BatchTask


def continuation_value(task: BatchTask, budget: int) -> float:
    """Proxy ψ: expected success gain from budget tokens."""
    if budget <= 0:
        return 0.0
    tau = task.oracle_tokens * 0.35
    scale = max(80.0, task.oracle_tokens * 0.5)
    surge = 1.0 - math.exp(-max(0.0, budget - tau) / scale)
    prior = min(0.95, 0.50 + 0.35 / max(task.difficulty, 0.5))
    return prior * surge


def marginal_psi(task: BatchTask, budget: int, *, delta: int = 40) -> float:
    return continuation_value(task, budget + delta) - continuation_value(task, budget)


def gittins_budget(task: BatchTask, lam: float, global_budget: int) -> int:
    """Budget where marginal ψ drops below λ; 0 if never clears λ at min budget."""
    min_b = 40
    if marginal_psi(task, min_b) < lam:
        return 0
    best = min_b
    for b in range(min_b, global_budget + 1, 40):
        if marginal_psi(task, b) >= lam:
            best = b
        else:
            break
    return min(best, global_budget)


@dataclass
class ReFORCOfficialAllocator:
    """Gittins-index bisection: Σ b_i(λ) ≤ B_total."""

    min_per_task: int = 40
    lambda_cost: float = 0.002
    max_iters: int = 32

    def allocate(self, tasks: list[BatchTask], global_budget: int) -> dict[str, int]:
        if not tasks:
            return {}
        n = len(tasks)
        if global_budget < self.min_per_task * n:
            base = max(1, global_budget // n)
            return {t.task_id: base for t in tasks}

        max_mu = max(marginal_psi(t, self.min_per_task) for t in tasks)
        lo, hi = self.lambda_cost, max(max_mu, self.lambda_cost * 2)
        best: dict[str, int] | None = None
        for _ in range(self.max_iters):
            mid = (lo + hi) / 2.0
            raw = {t.task_id: gittins_budget(t, mid, global_budget) for t in tasks}
            alloc = _normalize_to_budget(raw, global_budget, 0)
            if sum(alloc.values()) > global_budget:
                lo = mid
            else:
                hi = mid
                best = alloc

        if best is None:
            base = max(1, global_budget // n)
            return {t.task_id: base for t in tasks}
        return _normalize_to_budget(best, global_budget, self.min_per_task)


__all__ = ["ReFORCOfficialAllocator", "continuation_value", "gittins_budget"]
