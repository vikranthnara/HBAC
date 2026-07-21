"""Level-1 batch rewards optimized for Pass@1 + budget compliance (Phase 3a)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from hbac.training.batch_curriculum import TrainingBatch
from hbac.training.batch_rollout import TaskRolloutResult
from hbac.training.scarcity import HARD_BENCHMARKS


def domain_allocation_variance(alloc: dict[str, int], batch: TrainingBatch) -> float:
    """Variance of mean per-task budget across benchmarks (mode-collapse detector)."""
    by_bench: dict[str, list[int]] = defaultdict(list)
    for task in batch.tasks:
        by_bench[task.benchmark].append(alloc.get(task.task_id, 0))
    if len(by_bench) < 2:
        return 0.0
    means = [float(np.mean(v)) for v in by_bench.values()]
    return float(np.var(means))


def batch_violation_count(
    results: list[TaskRolloutResult],
    *,
    global_budget: int,
) -> tuple[int, int]:
    """Return (per_task_violations, batch_level_violation)."""
    task_viol = sum(1 for r in results if r.tokens_used > r.budget)
    batch_viol = int(sum(r.tokens_used for r in results) > global_budget)
    return task_viol, batch_viol


def starvation_rate(
    alloc: dict[str, int],
    batch: TrainingBatch,
    *,
    hard_min_frac: float = 0.15,
) -> float:
    """Fraction of hard benchmarks (SWE/LCB) below fair minimum budget."""
    hard = [t for t in batch.tasks if t.benchmark in HARD_BENCHMARKS]
    if not hard:
        return 0.0
    uniform = max(1, batch.global_budget // max(len(batch.tasks), 1))
    floor = max(1, int(uniform * hard_min_frac))
    starved = sum(1 for t in hard if alloc.get(t.task_id, 0) < floor)
    return starved / len(hard)


def l1_schema_reward(
    results: list[TaskRolloutResult],
    batch: TrainingBatch,
    alloc: dict[str, int],
    *,
    violation_penalty: float = 5.0,
    diversity_bonus: float = 0.15,
    parse_penalty: float = 0.0,
    starvation_penalty: float = 0.0,
    hard_min_frac: float = 0.15,
) -> float:
    """
    GRPO reward: Pass@1 primary, heavy violation penalty, domain diversity bonus.

    R = mean(S_i) - λ_v * violations - λ_p * mean(parse_fail_i) + β * Var_b(mean budget_b)
    """
    n = max(len(results), 1)
    pass_rate = sum(r.success for r in results) / n
    task_viol, batch_viol = batch_violation_count(results, global_budget=batch.global_budget)
    viol_rate = (task_viol + batch_viol) / (n + 1)
    parse_rate = sum(r.parse_failures for r in results) / n
    div = domain_allocation_variance(alloc, batch)
    starve = starvation_rate(alloc, batch, hard_min_frac=hard_min_frac)
    return (
        pass_rate
        - violation_penalty * viol_rate
        - parse_penalty * parse_rate
        - starvation_penalty * starve
        + diversity_bonus * (div / 1000.0)
    )


def compliant_pass_at_1(results: list[TaskRolloutResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.success and not r.budget_violated) / len(results)
