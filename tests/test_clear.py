from __future__ import annotations

from hbac.baselines.clear import CLEARAllocator, marginal_utility, surge_utility
from hbac.training.batch_curriculum import BatchTask


def _task(tid: str, oracle: int, diff: float = 1.0) -> BatchTask:
    return BatchTask(task_id=tid, benchmark="livecodebench", oracle_tokens=oracle, difficulty=diff)


class TestCLEARAllocator:
    def test_surge_monotone_in_budget(self):
        t = _task("a", 2000)
        u1 = surge_utility(t, 500, 10_000)
        u2 = surge_utility(t, 1500, 10_000)
        assert u2 >= u1

    def test_marginal_positive_for_hard_task(self):
        t = _task("hard", 4000, diff=1.5)
        assert marginal_utility(t, 200, 20_000) >= 0

    def test_allocate_respects_global_budget(self):
        tasks = [_task("a", 1000), _task("b", 3000), _task("c", 500)]
        alloc = CLEARAllocator().allocate(tasks, global_budget=2000)
        assert sum(alloc.values()) <= 2000
        assert set(alloc.keys()) == {"a", "b", "c"}

    def test_rational_abandonment_under_tight_budget(self):
        tasks = [_task("easy", 200, 0.8), _task("hard", 8000, 2.0)]
        alloc = CLEARAllocator(min_per_task=50).allocate(tasks, global_budget=400)
        assert sum(alloc.values()) <= 400
        assert alloc["hard"] == 0 or alloc["easy"] >= alloc.get("hard", 0)

    def test_non_uniform_vs_equal_split(self):
        tasks = [
            _task("easy", 300, 0.7),
            _task("hard", 6000, 1.8),
            _task("med", 2000, 1.1),
        ]
        clear_alloc = CLEARAllocator().allocate(tasks, global_budget=5000)
        uniform = {t.task_id: 5000 // 3 for t in tasks}
        assert max(clear_alloc.values()) - min(clear_alloc.values()) > max(uniform.values()) - min(
            uniform.values()
        )
