"""Re-FORC baseline — reward forecasting early stop [A3, arXiv:2511.02130].

Phase 1 uses HeuristicForecaster (Tier B proxy). LearnedForecaster requires
Beta adapter checkpoint from the original paper. Stopping rule follows
J = ψ - λC threshold inspired by Re-FORC §1. See Research Plan §9.1.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

from hbac.baselines.base import BaseRunner, RunnerConfig
from hbac.core.config import ReFORCConfig
from hbac.core.types import Observation


class RewardForecaster(ABC):
    @abstractmethod
    def predict(self, obs: Observation, turn: int, tokens_so_far: int) -> float:
        """Predict ψ(0 | context) ≈ expected future success in [0, 1]."""


class HeuristicForecaster(RewardForecaster):
    """Logistic-style forecaster on cheap features."""

    def __init__(self, lambda_cost: float = 0.001) -> None:
        self.lambda_cost = lambda_cost

    def predict(self, obs: Observation, turn: int, tokens_so_far: int) -> float:
        feedback = obs.env_feedback.lower()
        test_pass = 1.0 if "passed" in feedback or "success" in feedback else 0.0
        test_fail = 1.0 if "fail" in feedback or "error" in feedback else 0.0

        # Difficulty proxy: longer history and later turns reduce confidence
        history_len = sum(len(m.get("content", "")) for m in obs.history)
        turn_factor = 1.0 / (1.0 + 0.15 * turn)
        length_factor = 1.0 / (1.0 + history_len / 10000)

        logit = (
            -1.5
            + 2.5 * test_pass
            - 1.5 * test_fail
            + 0.5 * turn_factor
            + 0.3 * length_factor
            - self.lambda_cost * tokens_so_far / 1000
        )
        return 1.0 / (1.0 + math.exp(-logit))


class LearnedForecaster(RewardForecaster):
    """Placeholder for Re-FORC Beta-distribution adapter checkpoint."""

    def __init__(self, checkpoint_path: str, lambda_cost: float = 0.001) -> None:
        self.checkpoint_path = checkpoint_path
        self._fallback = HeuristicForecaster(lambda_cost)

    def predict(self, obs: Observation, turn: int, tokens_so_far: int) -> float:
        return self._fallback.predict(obs, turn, tokens_so_far)


def build_forecaster(config: ReFORCConfig) -> RewardForecaster:
    if config.mode == "learned":
        if not config.checkpoint_path:
            raise ValueError("Re-FORC learned mode requires checkpoint_path")
        return LearnedForecaster(config.checkpoint_path, config.lambda_cost)
    return HeuristicForecaster(config.lambda_cost)


class ReFORCRunner(BaseRunner):
    name = "ref_orc"

    def __init__(
        self,
        llm,
        config: RunnerConfig | None = None,
        reforc_config: ReFORCConfig | None = None,
    ) -> None:
        super().__init__(llm, config)
        self.reforc_config = reforc_config or ReFORCConfig()
        self.forecaster = build_forecaster(self.reforc_config)
        self._tokens_so_far = 0
        self._best_reward = 0.0

    def max_tokens_for_step(self, obs: Observation, turn: int) -> int:
        return self.config.max_tokens_per_step

    def should_stop_early(
        self, obs: Observation, turn: int, llm_text: str, step_tokens: int = 0
    ) -> bool:
        self._tokens_so_far += step_tokens
        psi = self.forecaster.predict(obs, turn, self._tokens_so_far)
        self._best_reward = max(self._best_reward, psi)

        # Re-FORC-stopping: stop if near success or marginal value below threshold
        if psi >= 0.95:
            return True

        expected_future_cost = 512 * self.reforc_config.lambda_cost
        marginal_value = psi - expected_future_cost
        if turn > 0 and marginal_value < self.reforc_config.stop_threshold:
            return True

        return False

    def run_episode(self, env, system_prompt: str, task_id: str):
        self._tokens_so_far = 0
        self._best_reward = 0.0
        return super().run_episode(env, system_prompt, task_id)
