"""Tests for Phase 3a L1 batch reward and gate thresholds."""

from __future__ import annotations

from pathlib import Path

from hbac.training.batch_curriculum import sample_batch
from hbac.training.batch_rollout import TaskRolloutResult
from hbac.training.l1_batch_reward import domain_allocation_variance, l1_schema_reward
from hbac.training.level1 import Level1Policy


class TestL1BatchReward:
    def test_domain_variance_mixed_batch(self):
        batch = sample_batch(Path("data/oracles"), seed=7)
        l1 = Level1Policy()
        alloc_uniform = l1.allocate_schema(batch, 0)
        alloc_code = l1.allocate_schema(batch, 1)
        var_u = domain_allocation_variance(alloc_uniform, batch)
        var_c = domain_allocation_variance(alloc_code, batch)
        assert var_c >= var_u or var_c > 0

    def test_schema_reward_penalizes_violations(self):
        batch = sample_batch(Path("data/oracles"), seed=1)
        ok = [
            TaskRolloutResult("t1", "swe_bench", True, 50, 100, 1.0, False),
            TaskRolloutResult("t2", "toolbench", True, 50, 100, 1.0, False),
        ]
        bad = [
            TaskRolloutResult("t1", "swe_bench", False, 150, 100, 0.0, True),
            TaskRolloutResult("t2", "toolbench", True, 50, 100, 1.0, False),
        ]
        alloc = {"t1": 100, "t2": 100}
        assert l1_schema_reward(ok, batch, alloc) > l1_schema_reward(bad, batch, alloc)
