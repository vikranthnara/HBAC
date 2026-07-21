"""Heuristic batch allocators for fair comparison (addresses reviewer SJF / type-prior baselines)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hbac.baselines.clear import _normalize_to_budget
from hbac.baselines.zebra import allocation_variance
from hbac.training.batch_curriculum import BatchTask
from hbac.training.scarcity import HARD_BENCHMARKS, TOOL_BENCHMARKS


@dataclass
class SJFAllocator:
    """
    Shortest-job-first: allocate budget inversely proportional to oracle length.
    Favors short (tool) tasks — upper bound on task-dropping heuristics.
    """

    min_per_task: int = 1

    def allocate(self, tasks: list[BatchTask], global_budget: int) -> dict[str, int]:
        if not tasks:
            return {}
        inv = np.array([1.0 / max(t.oracle_tokens, 1) for t in tasks], dtype=float)
        inv /= inv.sum()
        raw = {t.task_id: max(self.min_per_task, int(global_budget * w)) for t, w in zip(tasks, inv)}
        return _normalize_to_budget(raw, global_budget, self.min_per_task)


@dataclass
class TypePriorAllocator:
    """
    Type-based prior: minimal budget on hard benchmarks (SWE/LCB), rest to tool tasks.
    ~20-line heuristic the reviewer requested; no RL.
    """

    min_hard: int = 1

    def allocate(self, tasks: list[BatchTask], global_budget: int) -> dict[str, int]:
        if not tasks:
            return {}
        receivers = [t for t in tasks if t.benchmark in TOOL_BENCHMARKS]
        donors = [t for t in tasks if t.benchmark in HARD_BENCHMARKS]
        if not receivers:
            base = max(1, global_budget // len(tasks))
            return {t.task_id: base for t in tasks}
        raw = {t.task_id: self.min_hard for t in donors}
        pool = global_budget - sum(raw.values())
        per = max(1, pool // len(receivers)) if receivers else 0
        for t in receivers:
            raw[t.task_id] = raw.get(t.task_id, 0) + per
        return _normalize_to_budget(raw, global_budget, 1)


@dataclass
class DifficultyInverseAllocator:
    """Budget proportional to 1/difficulty — favors easy tasks under scarcity."""

    min_per_task: int = 1

    def allocate(self, tasks: list[BatchTask], global_budget: int) -> dict[str, int]:
        if not tasks:
            return {}
        w = np.array([1.0 / max(t.difficulty, 0.1) for t in tasks], dtype=float)
        w /= w.sum()
        raw = {t.task_id: max(self.min_per_task, int(global_budget * x)) for t, x in zip(tasks, w)}
        return _normalize_to_budget(raw, global_budget, self.min_per_task)


@dataclass
class BatchTABProxyAllocator:
    """
    Batch-level TAB proxy: turn-budget philosophy mapped to per-task caps.
    Higher difficulty → more per-task budget (math-heavy problems need turns).
    """

    min_per_task: int = 40

    def allocate(self, tasks: list[BatchTask], global_budget: int) -> dict[str, int]:
        if not tasks:
            return {}
        w = np.array([0.5 + t.difficulty for t in tasks], dtype=float)
        w /= w.sum()
        raw = {t.task_id: max(self.min_per_task, int(global_budget * x)) for t, x in zip(tasks, w)}
        return _normalize_to_budget(raw, global_budget, self.min_per_task)


@dataclass
class BatchReFORCProxyAllocator:
    """
    Batch-level Re-FORC proxy: low budget when marginal oracle utility is flat.
    Uses oracle_tokens as proxy for continuation value.
    """

    min_per_task: int = 40
    stop_threshold: float = 0.15

    def allocate(self, tasks: list[BatchTask], global_budget: int) -> dict[str, int]:
        if not tasks:
            return {}
        raw: dict[str, int] = {}
        for t in tasks:
            marginal = 1.0 / max(t.oracle_tokens, 1) * (1.0 - t.difficulty * 0.1)
            if marginal < self.stop_threshold:
                raw[t.task_id] = self.min_per_task
            else:
                raw[t.task_id] = max(self.min_per_task, int(global_budget * marginal / len(tasks)))
        return _normalize_to_budget(raw, global_budget, self.min_per_task)


__all__ = [
    "SJFAllocator",
    "TypePriorAllocator",
    "DifficultyInverseAllocator",
    "BatchTABProxyAllocator",
    "BatchReFORCProxyAllocator",
    "allocation_variance",
]
