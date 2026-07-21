from __future__ import annotations

from hbac.baselines.clear_official import CLEAROfficialAllocator, lambert_allocation
from hbac.baselines.zebra_official import ZEBRAOfficialAllocator, waterfill_allocation
from hbac.training.batch_curriculum import BatchTask


def _task(tid: str, oracle: int, diff: float = 1.0) -> BatchTask:
    return BatchTask(task_id=tid, benchmark="livecodebench", oracle_tokens=oracle, difficulty=diff)


class TestOfficialBaselines:
    def test_lambert_allocation_respects_budget(self):
        t = _task("a", 2000, 1.2)
        b = lambert_allocation(t, lam=0.01, global_budget=3000)
        assert 0 <= b <= 3000

    def test_clear_official_sums_to_budget(self):
        tasks = [_task("a", 800, 0.8), _task("b", 3000, 1.5)]
        alloc = CLEAROfficialAllocator(min_per_task=40).allocate(tasks, global_budget=1200)
        assert sum(alloc.values()) <= 1200

    def test_zebra_waterfill_monotone_in_lambda(self):
        t = _task("hard", 4000, 1.8)
        x_lo = waterfill_allocation(t, 0.001)
        x_hi = waterfill_allocation(t, 0.1)
        assert x_lo >= x_hi

    def test_zebra_official_non_empty(self):
        tasks = [_task("a", 500, 1.0), _task("b", 2000, 1.2)]
        alloc = ZEBRAOfficialAllocator(min_per_task=40).allocate(tasks, global_budget=800)
        assert sum(alloc.values()) <= 800
        assert len(alloc) == 2
