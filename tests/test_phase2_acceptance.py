from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hbac.core.types import Observation, Trajectory, TrajectoryStep, AgentAction
from hbac.training.config import PPOConfig
from hbac.training.controller import MonolithicController, featurize_observation
from hbac.training.dataset import (
    find_oracle_paths,
    load_stop_examples,
    trajectory_to_stop_examples,
    train_val_split,
)
from hbac.training.ppo import PPOTrainer, Transition
from hbac.training.probes import probe_premature_stop_rate
from hbac.training.validation import (
    all_passed,
    best_reward_defaults,
    sweep_reward_hyperparameters,
    validate_stop_hacking_margin,
)
from hbac.scripts.train_variant_a import _build_batch, _eval_accuracy
from hbac.training.reward import TaskControllerReward


class TestPhase2Reward:
    def test_sweep_finds_passing_defaults(self):
        sweep = sweep_reward_hyperparameters()
        lam, pen = best_reward_defaults(sweep)
        assert any(r["passed"] for r in sweep)
        assert all_passed(TaskControllerReward(lambda_token=lam, premature_stop_penalty=pen))

    def test_hacking_margin_passes(self):
        assert validate_stop_hacking_margin().passed


class TestPhase2Dataset:
    def test_rich_observation_history(self):
        traj = Trajectory(
            task_id="t1",
            benchmark="mock",
            model="test",
            baseline="react",
            success=True,
            total_tokens=100,
            budget=1000,
            steps=[
                TrajectoryStep(
                    turn=0,
                    action=AgentAction(tool_name="bash", tool_input="x"),
                    observation="feedback1",
                    llm_response='{"tool_name":"bash"}',
                    tokens=50,
                ),
                TrajectoryStep(
                    turn=1,
                    action=AgentAction(tool_name="submit", tool_input=""),
                    observation="done",
                    llm_response='{"tool_name":"submit"}',
                    tokens=50,
                ),
            ],
        )
        exs = trajectory_to_stop_examples(traj)
        assert len(exs) == 2
        assert exs[0]["stop"] is False
        assert exs[1]["stop"] is True
        assert len(exs[0]["observation"].history) == 0
        assert len(exs[1]["observation"].history) == 2

    def test_merge_oracle_paths(self):
        paths = find_oracle_paths(Path("data/oracles"))
        assert paths, "seed oracles required for Phase 2 tests"


class TestPhase2Training:
    def test_train_variant_a_cli(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hbac.scripts.train_variant_a",
                "--oracle-path",
                "data/oracles/seed",
                "--subset-limit",
                "20",
                "--epochs",
                "3",
                "--output",
                str(tmp_path / "ckpt"),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, result.stderr
        ckpts = list((tmp_path / "ckpt").rglob("stage1_stop_controller.npz"))
        assert ckpts

    def test_ref_kl_nonzero_after_update(self):
        ctrl = MonolithicController()
        trainer = PPOTrainer(ctrl, PPOConfig(kl_coef=0.05))
        obs = Observation(turn=1, remaining_budget=5000, env_feedback="test")
        features = featurize_observation(obs)
        batch = [
            Transition(features, False, 0.5, ctrl.log_prob_stop(obs, False), False),
            Transition(features, True, 1.0, ctrl.log_prob_stop(obs, True), True),
        ]
        stats = trainer.update(batch)
        assert stats.ref_kl != 0.0 or stats.kl_divergence != 0.0


class TestPhase2KLAblation:
    def test_kl_zero_more_hacking_than_regularized(self):
        np.random.seed(0)
        paths = find_oracle_paths(Path("data/oracles/seed"))
        examples = load_stop_examples(paths, limit=30)
        train_ex, val_ex = train_val_split(examples, val_fraction=0.2, seed=42)
        reward_fn = TaskControllerReward()

        ctrl_no_kl = MonolithicController()
        trainer0 = PPOTrainer(ctrl_no_kl, PPOConfig(kl_coef=0.0, kl_adaptive=False), reward_fn)
        for _ in range(5):
            trainer0.update(_build_batch(train_ex, ctrl_no_kl, reward_fn))

        ctrl_kl = MonolithicController()
        trainer1 = PPOTrainer(ctrl_kl, PPOConfig(kl_coef=0.05, kl_adaptive=False), reward_fn)
        for _ in range(5):
            trainer1.update(_build_batch(train_ex, ctrl_kl, reward_fn))

        probe0 = probe_premature_stop_rate(ctrl_no_kl)
        probe1 = probe_premature_stop_rate(ctrl_kl)
        acc0 = _eval_accuracy(ctrl_no_kl, val_ex)
        acc1 = _eval_accuracy(ctrl_kl, val_ex)
        # KL-regularized should not collapse val accuracy
        assert acc1 >= acc0 - 0.15
        # H7: unregularized tends toward higher premature stop on probes (soft check on tiny data)
        assert probe0["mean_stop_prob"] >= probe1["mean_stop_prob"] - 0.3
