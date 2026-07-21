"""ZEBRA-style zero-shot water-filling batch allocator (Tier B proxy) [A5]."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hbac.baselines.clear import _normalize_to_budget, allocation_variance
from hbac.training.batch_curriculum import BatchTask


def _task_weight(task: BatchTask) -> float:
    """Proxy priority: higher difficulty + more oracle tokens → more budget."""
    return max(1.0, task.oracle_tokens) * max(0.5, task.difficulty)


@dataclass
class ZEBRAAllocator:
    """Lagrangian water-filling: allocate proportional to weights, project to budget."""

    min_per_task: int = 50

    def allocate(self, tasks: list[BatchTask], global_budget: int) -> dict[str, int]:
        if not tasks:
            return {}
        n = len(tasks)
        if global_budget < self.min_per_task * n:
            base = max(1, global_budget // n)
            return {t.task_id: base for t in tasks}

        weights = np.array([_task_weight(t) for t in tasks], dtype=float)
        weights /= weights.sum()
        raw = {
            t.task_id: max(self.min_per_task, int(global_budget * w))
            for t, w in zip(tasks, weights)
        }
        return _normalize_to_budget(raw, global_budget, self.min_per_task)


__all__ = ["ZEBRAAllocator", "allocation_variance"]
