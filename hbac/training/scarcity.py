"""Scarcity-aware allocation post-processing (inference-time, no retrain)."""

from __future__ import annotations

from hbac.training.batch_curriculum import BatchTask, TrainingBatch


TOOL_BENCHMARKS = frozenset({"tau_bench", "toolbench", "mock"})
HARD_BENCHMARKS = frozenset({"swe_bench", "livecodebench"})


def scarcity_boost_alloc(
    alloc: dict[str, int],
    batch: TrainingBatch,
    *,
    scarcity_threshold: int = 450,
    shift_fraction: float = 0.15,
    swe_min_reserve: float = 0.5,
) -> dict[str, int]:
    """
    Under tight per-task caps, shift budget from low-priority (SWE) tasks toward
    tool-heavy tasks that need multi-step chains (τ, toolbench).

    Only activates when mean allocation < scarcity_threshold tokens/task.
    swe_min_reserve: fraction of each donor's budget to keep (avoids SWE parse collapse).
    """
    if not alloc or not batch.tasks:
        return alloc
    n = len(batch.tasks)
    mean_alloc = sum(alloc.values()) / n
    if mean_alloc >= scarcity_threshold:
        return dict(alloc)

    by_id = {t.task_id: t for t in batch.tasks}
    donors: list[str] = []
    receivers: list[str] = []
    for tid, budget in alloc.items():
        task = by_id.get(tid)
        if task is None:
            continue
        if task.benchmark in HARD_BENCHMARKS and budget > 1:
            donors.append(tid)
        elif task.benchmark in TOOL_BENCHMARKS:
            receivers.append(tid)

    if not donors or not receivers:
        return dict(alloc)

    out = dict(alloc)
    pool = 0
    per_donor = max(1, int(mean_alloc * shift_fraction))
    reserve = max(0.0, min(1.0, swe_min_reserve))
    for tid in donors:
        floor = max(1, int(out[tid] * reserve))
        take = min(per_donor, max(0, out[tid] - floor))
        out[tid] -= take
        pool += take

    if pool <= 0:
        return out

    per_recv = max(1, pool // len(receivers))
    remainder = pool
    for i, tid in enumerate(receivers):
        add = min(per_recv, remainder) if i < len(receivers) - 1 else remainder
        out[tid] += add
        remainder -= add

    # Re-project to global budget
    total = sum(out.values())
    if total > batch.global_budget:
        scale = batch.global_budget / total
        out = {k: max(1, int(v * scale)) for k, v in out.items()}
        while sum(out.values()) > batch.global_budget:
            k_max = max(out, key=lambda x: out[x])
            if out[k_max] <= 1:
                break
            out[k_max] -= 1

    return out


def roi_skip_alloc(
    alloc: dict[str, int],
    batch: TrainingBatch,
    *,
    floor_threshold: int = 350,
) -> dict[str, int]:
    """
    Under extreme scarcity (mean alloc below floor_threshold), zero out hard
    benchmarks (SWE/LCB) and redistribute budget to tool-heavy tasks (D14).
    """
    if not alloc or not batch.tasks:
        return alloc
    n = len(batch.tasks)
    mean_alloc = sum(alloc.values()) / n
    if mean_alloc >= floor_threshold:
        return dict(alloc)

    by_id = {t.task_id: t for t in batch.tasks}
    receivers = [
        tid
        for tid, budget in alloc.items()
        if (task := by_id.get(tid)) and task.benchmark in TOOL_BENCHMARKS and budget >= 1
    ]
    if not receivers:
        return dict(alloc)

    out = dict(alloc)
    pool = 0
    for tid, budget in list(out.items()):
        task = by_id.get(tid)
        if task is None or task.benchmark not in HARD_BENCHMARKS:
            continue
        pool += max(0, budget - 1)
        out[tid] = 1

    if pool <= 0:
        return out

    per_recv = max(1, pool // len(receivers))
    remainder = pool
    for i, tid in enumerate(receivers):
        add = min(per_recv, remainder) if i < len(receivers) - 1 else remainder
        out[tid] += add
        remainder -= add

    total = sum(out.values())
    if total > batch.global_budget:
        scale = batch.global_budget / total
        out = {k: max(1, int(v * scale)) for k, v in out.items()}
        while sum(out.values()) > batch.global_budget:
            k_max = max(out, key=lambda x: out[x])
            if out[k_max] <= 1:
                break
            out[k_max] -= 1

    return out


def fairness_reserve_alloc(
    alloc: dict[str, int],
    batch: TrainingBatch,
    *,
    hard_min_frac: float = 0.12,
    uniform_floor: bool = True,
) -> dict[str, int]:
    """
    Guarantee each hard benchmark (SWE/LCB) receives at least hard_min_frac of
    uniform per-task share. Beats type-prior starvation while preserving HBAC
    differentiation on tool tasks (D17 fairness constraint).
    """
    if not alloc or not batch.tasks:
        return alloc
    n = len(batch.tasks)
    uniform_share = max(1, batch.global_budget // n)
    min_hard = max(1, int(uniform_share * hard_min_frac))
    if uniform_floor:
        min_hard = max(min_hard, 40 if uniform_share >= 40 else 1)

    by_id = {t.task_id: t for t in batch.tasks}
    out = dict(alloc)
    for tid, budget in list(out.items()):
        task = by_id.get(tid)
        if task and task.benchmark in HARD_BENCHMARKS:
            out[tid] = max(budget, min_hard)

    total = sum(out.values())
    if total <= batch.global_budget:
        return out

    # Trim from tool receivers proportionally
    receivers = [
        tid
        for tid, b in out.items()
        if (t := by_id.get(tid)) and t.benchmark in TOOL_BENCHMARKS and b > 1
    ]
    overflow = total - batch.global_budget
    if not receivers:
        scale = batch.global_budget / total
        return {k: max(1, int(v * scale)) for k, v in out.items()}

    per_trim = max(1, overflow // len(receivers))
    for tid in receivers:
        take = min(per_trim, max(0, out[tid] - 1))
        out[tid] -= take
        overflow -= take
        if overflow <= 0:
            break

    while sum(out.values()) > batch.global_budget:
        k_max = max(
            (tid for tid in receivers if out[tid] > 1),
            key=lambda k: out[k],
            default=None,
        )
        if k_max is None:
            break
        out[k_max] -= 1

    return out


__all__ = [
    "scarcity_boost_alloc",
    "roi_skip_alloc",
    "fairness_reserve_alloc",
    "TOOL_BENCHMARKS",
    "HARD_BENCHMARKS",
]
