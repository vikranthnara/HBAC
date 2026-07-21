from __future__ import annotations

from hbac.training.batch_curriculum import BatchTask, TrainingBatch
from hbac.training.scarcity import fairness_reserve_alloc


def _batch(tasks: list[BatchTask], budget: int = 400) -> TrainingBatch:
    return TrainingBatch(
        batch_id="b1",
        tasks=tasks,
        global_budget=budget,
        oracle_token_sum=sum(t.oracle_tokens for t in tasks),
        budget_fraction=0.4,
    )


class TestFairnessReserve:
    def test_hard_tasks_get_minimum(self):
        tasks = [
            BatchTask("swe", "swe_bench", 3000, 1.5),
            BatchTask("tool", "toolbench", 200, 0.5),
        ]
        batch = _batch(tasks, budget=400)
        alloc = {"swe": 1, "tool": 399}
        out = fairness_reserve_alloc(alloc, batch, hard_min_frac=0.15)
        assert out["swe"] > 1
        assert sum(out.values()) <= 400
