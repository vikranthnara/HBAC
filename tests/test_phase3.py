"""Phase 3 training and full pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hbac.training.batch_curriculum import generate_curriculum_batches, sample_batch
from hbac.training.batch_rollout import rollout_batch_schema
from hbac.training.controller import MonolithicController
from hbac.training.credit import compute_counterfactual_credits
from hbac.training.grpo import GRPOTrainer, L2GRPOTrainer, group_advantages
from hbac.training.level1 import Level1Allocator, Level1Policy
from hbac.training.oracle_replay import OracleIndex, rollout_task_with_oracle
from hbac.training.phase3_pipeline import Stage3Config, run_full_phase3
from hbac.training.utility_net import UtilityNetwork
from hbac.training.dataset import load_stop_examples, find_oracle_paths


class TestPhase3:
    def test_group_advantages_normalized(self):
        adv, skipped = group_advantages([1.0, 0.0, 0.5, 0.5])
        assert not skipped
        assert abs(float(adv.mean())) < 1e-6

    def test_group_advantages_starvation(self):
        adv, skipped = group_advantages([0.0, 0.0, 0.0])
        assert skipped

    def test_sample_batch(self):
        batch = sample_batch(Path("data/oracles"), seed=0)
        assert batch.tasks
        assert batch.global_budget > 0

    def test_level1_policy_allocations_sum(self):
        batch = sample_batch(Path("data/oracles"), seed=1)
        policy = Level1Policy()
        for sid in range(policy.num_schemas):
            alloc = policy.allocate_schema(batch, sid)
            assert sum(alloc.values()) <= batch.global_budget

    def test_grpo_l1_update(self):
        batch = sample_batch(Path("data/oracles"), seed=2)
        policy = Level1Policy()
        trainer = GRPOTrainer(policy)
        schema_ids = [0, 1, 2, 3]
        rewards = [1.0, 0.2, 0.5, 0.8]
        stats = trainer.update_l1(batch, schema_ids, rewards)
        assert not stats.skipped

    def test_batch_rollout(self):
        batch = sample_batch(Path("data/oracles"), seed=3)
        l2 = MonolithicController()
        alloc = Level1Allocator(batch.global_budget).allocate(batch.task_ids)
        result = rollout_batch_schema(batch, alloc, l2)
        assert len(result.task_results) == len(batch.tasks)

    def test_utility_allocate(self):
        batch = sample_batch(Path("data/oracles"), seed=4)
        net = UtilityNetwork()
        alloc = net.allocate_greedy(batch.tasks, batch.global_budget, min_per_task=20)
        assert len(alloc) >= 1
        assert sum(alloc.values()) <= batch.global_budget + 1

    def test_generate_curriculum(self):
        batches = generate_curriculum_batches(Path("data/oracles"), num_batches=3, seed=0)
        assert len(batches) == 3
        fracs = {b.budget_fraction for b in batches}
        assert fracs  # 90/75/60 cycle

    def test_train_variant_b_cli(self):
        import subprocess
        import sys

        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "hbac.scripts.train_variant_b",
                "--num-batches",
                "2",
                "--epochs",
                "1",
                "--grpo-groups",
                "4",
                "--output",
                "checkpoints/variant_b/test_run",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert r.returncode == 0, r.stderr

    def test_oracle_replay_rollout(self):
        batch = sample_batch(Path("data/oracles"), seed=5)
        idx = OracleIndex(Path("data/oracles"))
        l2 = MonolithicController()
        task = batch.tasks[0]
        alloc = Level1Allocator(batch.global_budget).allocate(batch.task_ids)
        result = rollout_task_with_oracle(task, alloc[task.task_id], l2, idx)
        assert result.tokens_used >= 0

    def test_counterfactual_credit(self):
        batch = sample_batch(Path("data/oracles"), seed=6)
        l2 = MonolithicController()
        idx = OracleIndex(Path("data/oracles"))
        alloc = Level1Allocator(batch.global_budget).allocate(batch.task_ids)
        rollout = rollout_batch_schema(batch, alloc, l2)
        credits = compute_counterfactual_credits(batch, rollout, l2, idx)
        assert len(credits) == len(batch.tasks)

    def test_l2_grpo_update(self):
        paths = find_oracle_paths(Path("data/oracles"))
        ex = load_stop_examples(paths, limit=20)
        if not ex:
            pytest.skip("no examples")
        l2 = MonolithicController()
        trainer = L2GRPOTrainer(l2)
        stats = trainer.update_from_examples(ex)
        assert stats is not None

    def test_run_phase3_pipeline(self, tmp_path):
        report = run_full_phase3(
            Path("data/oracles"),
            Path("checkpoints/variant_a"),
            tmp_path / "phase3",
            Stage3Config(num_batches=4, epochs=2, grpo_groups=4),
            run_stage4=False,
            run_variant_a=True,
        )
        assert report.stage3_dir.is_dir()
        assert (report.stage3_dir / "level1_policy.npz").is_file()
        assert report.stage3_metrics.pass_at_1 >= 0.0

    def test_llm_grpo_trainer(self, tmp_path):
        from hbac.training.llm_grpo_trainer import load_sft_prompts, train_with_trl

        prompts = load_sft_prompts(Path("data/oracles"), limit=4)
        if not prompts:
            pytest.skip("no oracle prompts")
        log = train_with_trl(
            prompts,
            "gpt2",
            tmp_path / "llm",
            grpo_groups=2,
            epochs=1,
            max_samples=2,
        )
        assert log
        assert (tmp_path / "llm" / "model" / "adapter_config.json").is_file() or (
            tmp_path / "llm" / "model" / "config.json"
        ).is_file()
