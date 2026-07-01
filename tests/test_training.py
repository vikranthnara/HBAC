from __future__ import annotations

import numpy as np
import pytest

from hbac.training.config import PPOConfig
from hbac.training.controller import MonolithicController, featurize_observation
from hbac.training.ppo import PPOTrainer, Transition
from hbac.training.reward import TaskControllerReward
from hbac.training.validation import all_passed, run_all_validations
from hbac.core.types import Observation


class TestRewardValidation:
    def test_all_invariants_pass_default(self):
        assert all_passed()

    def test_premature_stop_beats_success_when_penalty_zero(self):
        r = TaskControllerReward(premature_stop_penalty=0.0)
        results = run_all_validations(r)
        premature = next(x for x in results if x.name == "success_dominates_premature_stop")
        # With zero penalty, success still wins because success=1 vs success=0
        assert premature.passed

    def test_high_lambda_penalizes_tokens(self):
        r = TaskControllerReward(lambda_token=0.01)
        low = r.terminal(success=True, tokens_used=1000, budget=5000, env_done=True)
        high = r.terminal(success=True, tokens_used=4000, budget=5000, env_done=True)
        assert low > high


class TestMonolithicController:
    def test_stop_prob_bounds(self):
        ctrl = MonolithicController()
        obs = Observation(turn=1, remaining_budget=5000)
        p = ctrl.stop_prob(obs)
        assert 0.0 <= p <= 1.0

    def test_featurize_dim(self):
        obs = Observation(turn=2, remaining_budget=1000, history=[{"role": "user", "content": "hi"}])
        assert featurize_observation(obs).shape == (7,)


class TestPPOTrainer:
    def test_update_returns_stats(self):
        ctrl = MonolithicController()
        trainer = PPOTrainer(ctrl, PPOConfig(kl_coef=0.05))
        obs = Observation(turn=1, remaining_budget=5000)
        features = featurize_observation(obs)
        lp = ctrl.log_prob_stop(obs, False)
        batch = [
            Transition(features, False, 0.5, lp, False),
            Transition(features, True, 0.8, ctrl.log_prob_stop(obs, True), True),
        ]
        stats = trainer.update(batch)
        assert stats.kl_coef > 0
        assert isinstance(stats.kl_divergence, float)

    def test_kl_coef_adapts(self):
        ctrl = MonolithicController()
        trainer = PPOTrainer(ctrl, PPOConfig(kl_coef=0.02, kl_adaptive=True, kl_target=0.01))
        initial = trainer._kl_coef
        trainer.adapt_kl_coef(0.05)  # high KL -> increase coef
        assert trainer._kl_coef > initial
