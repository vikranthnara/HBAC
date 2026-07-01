"""CLEAR (BL4) inference-only shadow-price batch allocator — Tier B proxy.

Grounding: Wan et al. (2026) CLEAR [A4] — shifted-surge utility U(q,b) and global
shadow price λ via bisection; rational abandonment when marginal utility < λ.

This is an engineering proxy using oracle metadata (tokens, difficulty) rather than
a trained utility-curve predictor from the CLEAR paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hbac.training.batch_curriculum import BatchTask


def surge_utility(task: BatchTask, budget: int, global_budget: int) -> float:
    """Proxy U(q, b): success prior × shifted-surge saturation."""
    if budget <= 0:
        return 0.0
    tau = task.oracle_tokens * (0.35 + 0.25 * task.difficulty)
    scale = max(tau * 0.45, 80.0)
    shifted = max(0.0, budget - tau * 0.25)
    surge = 1.0 - math.exp(-shifted / scale)
    prior = min(0.95, 0.45 + 0.35 / max(task.difficulty, 0.5))
    budget_penalty = 0.02 * (budget / max(global_budget, 1))
    return max(0.0, prior * surge - budget_penalty)


def marginal_utility(
    task: BatchTask,
    budget: int,
    global_budget: int,
    *,
    delta: int = 50,
) -> float:
    hi = budget + delta
    return surge_utility(task, hi, global_budget) - surge_utility(task, budget, global_budget)


def optimal_budget_at_lambda(
    task: BatchTask,
    lam: float,
    global_budget: int,
    *,
    min_budget: int = 50,
    step: int = 50,
) -> int:
    """Rational abandonment: return 0 when no budget level clears shadow price λ."""
    if marginal_utility(task, min_budget, global_budget, delta=step) < lam:
        return 0
    best = min_budget
    for budget in range(min_budget, global_budget + 1, step):
        if marginal_utility(task, budget, global_budget, delta=step) >= lam:
            best = budget
        else:
            break
    return min(best, global_budget)


@dataclass
class CLEARAllocator:
    """Inference-only CLEAR compose baseline: frozen L2 + shadow-price L1."""

    min_per_task: int = 50
    budget_step: int = 50
    max_bisect_iters: int = 32

    def _demand(self, tasks: list[BatchTask], global_budget: int, lam: float) -> int:
        return sum(
            optimal_budget_at_lambda(
                t,
                lam,
                global_budget,
                min_budget=self.min_per_task,
                step=self.budget_step,
            )
            for t in tasks
        )

    def allocate(self, tasks: list[BatchTask], global_budget: int) -> dict[str, int]:
        if not tasks:
            return {}
        n = len(tasks)
        if global_budget < self.min_per_task * n:
            base = max(1, global_budget // n)
            return {t.task_id: base for t in tasks}

        max_mu = max(
            marginal_utility(t, self.min_per_task, global_budget, delta=self.budget_step)
            for t in tasks
        )
        if max_mu <= 0:
            base = max(1, global_budget // n)
            return {t.task_id: base for t in tasks}

        lo_lam, hi_lam = 0.0, max_mu
        best_alloc: dict[str, int] | None = None
        for _ in range(self.max_bisect_iters):
            mid = (lo_lam + hi_lam) / 2.0
            raw = {
                t.task_id: optimal_budget_at_lambda(
                    t,
                    mid,
                    global_budget,
                    min_budget=self.min_per_task,
                    step=self.budget_step,
                )
                for t in tasks
            }
            alloc = _normalize_to_budget(raw, global_budget, self.min_per_task)
            demand = sum(alloc.values())
            if demand > global_budget:
                lo_lam = mid
            else:
                hi_lam = mid
                best_alloc = alloc

        if best_alloc is None:
            base = max(1, global_budget // n)
            return {t.task_id: base for t in tasks}
        return best_alloc


def _normalize_to_budget(alloc: dict[str, int], global_budget: int, min_per_task: int) -> dict[str, int]:
    """Project integer allocations to sum <= global_budget."""
    out = {k: max(0, int(v)) for k, v in alloc.items()}
    if not out:
        return out
    if sum(out.values()) == 0:
        base = max(1, global_budget // len(out))
        return {k: base for k in out}

    total = sum(out.values())
    if total <= global_budget:
        remainder = global_budget - total
        keys = [k for k, v in out.items() if v > 0] or list(out.keys())
        for i in range(remainder):
            out[keys[i % len(keys)]] += 1
        return out

    scale = global_budget / total
    out = {k: max(0, int(v * scale)) for k, v in out.items()}
    while sum(out.values()) > global_budget:
        k_max = max(out, key=lambda k: out[k])
        if out[k_max] <= 0:
            break
        out[k_max] -= 1
    deficit = global_budget - sum(out.values())
    keys = list(out.keys())
    for i in range(deficit):
        out[keys[i % len(keys)]] += 1
    return out


def allocation_variance(allocations: dict[str, int]) -> float:
    if not allocations:
        return 0.0
    return float(np.var(list(allocations.values())))
