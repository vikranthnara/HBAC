"""CLEAR (Lambert W) allocator — Tier-A faithful to Wan et al. ICML 2026 [A4].

Implements Eq. (5) shadow-price parity with bisection on λ (Algorithm 1).
Task parameters (τ, α, β) estimated from oracle metadata when curve predictor
is unavailable — same information budget as our Tier-B proxy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hbac.baselines.clear import _normalize_to_budget
from hbac.training.batch_curriculum import BatchTask


def _lambert_w0(z: float) -> float:
    """Principal branch Lambert W via Halley iteration (no scipy dep)."""
    if z <= 0:
        return 0.0
    if z > 700:
        return math.log(z)  # asymptotic W(z) ~ log(z) for large z
    w = math.log(max(z, 1e-300))
    if z > 1.0:
        w -= math.log(max(w, 1e-300))
    for _ in range(12):
        w = max(-1.0, min(w, 20.0))
        ew = math.exp(w)
        wew = w * ew
        f = wew - z
        df = ew * (w + 1.0)
        ddf = ew * (w + 2.0)
        denom = 2.0 * df * df - f * ddf
        if abs(denom) < 1e-12:
            break
        w -= (2.0 * f * df) / denom
    return w


def _estimate_curve_params(task: BatchTask) -> tuple[float, float, float]:
    """Proxy (τ, α, β) from oracle length and difficulty."""
    tau = max(40.0, task.oracle_tokens * (0.30 + 0.20 * task.difficulty))
    alpha = min(1.0, 0.55 + 0.40 / max(task.difficulty, 0.5))
    beta = max(0.002, 1.0 / max(tau * 0.55, 80.0))
    return tau, alpha, beta


def lambert_allocation(
    task: BatchTask,
    lam: float,
    *,
    global_budget: int,
) -> int:
    """Eq. (5): optimal tokens at shadow price λ with solvency check."""
    if lam <= 0:
        return min(global_budget, int(task.oracle_tokens * 1.1))
    tau, alpha, beta = _estimate_curve_params(task)
    z = lam / (alpha * beta * math.e)
    z = max(z, 1e-12)
    w = _lambert_w0(z)
    delta_t = (1.0 / beta) * (1.0 - w)
    t_star = tau + max(0.0, delta_t)
    # Solvency: abandon if utility <= cost
    utility = alpha * max(0.0, t_star - tau) * math.exp(-beta * max(0.0, t_star - tau))
    if utility <= lam * t_star or t_star <= tau:
        return 0
    return int(min(t_star, global_budget))


@dataclass
class CLEAROfficialAllocator:
    """Paper-faithful CLEAR (Lambert) with bisection on global shadow price."""

    min_per_task: int = 40
    max_bisect_iters: int = 40

    def allocate(self, tasks: list[BatchTask], global_budget: int) -> dict[str, int]:
        if not tasks:
            return {}
        n = len(tasks)
        if global_budget < self.min_per_task * n:
            base = max(1, global_budget // n)
            return {t.task_id: base for t in tasks}

        lo, hi = 0.0, 1.0
        best: dict[str, int] | None = None
        for _ in range(self.max_bisect_iters):
            mid = (lo + hi) / 2.0
            raw = {
                t.task_id: lambert_allocation(t, mid, global_budget=global_budget)
                for t in tasks
            }
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


__all__ = ["CLEAROfficialAllocator", "lambert_allocation"]
