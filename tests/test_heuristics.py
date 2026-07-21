from __future__ import annotations

from hbac.baselines.heuristics import (
    DifficultyInverseAllocator,
    SJFAllocator,
    TypePriorAllocator,
)
from hbac.training.batch_curriculum import BatchTask


def _task(tid: str, bench: str, oracle: int, diff: float = 1.0) -> BatchTask:
    return BatchTask(
        task_id=tid, benchmark=bench, oracle_tokens=oracle, difficulty=diff
    )


class TestHeuristicAllocators:
    def test_sjf_favors_short_tasks(self):
        tasks = [
            _task("short", "toolbench", 100, 0.5),
            _task("long", "swe_bench", 5000, 1.5),
        ]
        alloc = SJFAllocator(min_per_task=1).allocate(tasks, global_budget=200)
        assert sum(alloc.values()) <= 200
        assert alloc["short"] >= alloc["long"]

    def test_type_prior_starves_hard_benchmarks(self):
        tasks = [
            _task("swe", "swe_bench", 3000, 1.5),
            _task("tool", "toolbench", 200, 0.5),
        ]
        alloc = TypePriorAllocator(min_hard=1).allocate(tasks, global_budget=400)
        assert alloc["swe"] <= alloc["tool"]
        assert sum(alloc.values()) <= 400

    def test_difficulty_inverse_respects_budget(self):
        tasks = [
            _task("a", "mock", 500, 0.3),
            _task("b", "mock", 500, 2.0),
        ]
        alloc = DifficultyInverseAllocator().allocate(tasks, global_budget=300)
        assert sum(alloc.values()) <= 300
        assert alloc["a"] >= alloc["b"]
