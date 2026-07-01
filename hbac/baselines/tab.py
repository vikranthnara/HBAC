"""TAB baseline — Turn-Adaptive Budgets [A2, arXiv:2604.05164].

Phase 1 uses HeuristicTABPolicy (Tier B proxy). LearnedTABPolicy requires
GRPO-trained checkpoint from the original paper for paper-faithful evaluation.
See Research Plan §9.1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from hbac.baselines.base import BaseRunner, RunnerConfig
from hbac.core.config import TABConfig
from hbac.core.types import AgentAction, Observation


@dataclass
class TABState:
    allocated_budgets: list[int] = field(default_factory=list)
    global_budget: int = 50_000


class TABPolicy(ABC):
    @abstractmethod
    def allocate(self, obs: Observation, turn: int, global_budget: int, used: int) -> int: ...


class StaticTABPolicy(TABPolicy):
    def __init__(self, budget_per_turn: int = 2048) -> None:
        self.budget_per_turn = budget_per_turn

    def allocate(self, obs: Observation, turn: int, global_budget: int, used: int) -> int:
        return min(self.budget_per_turn, global_budget - used)


class HeuristicTABPolicy(TABPolicy):
    """Rule-based per-turn budget using difficulty proxies."""

    def __init__(self, budget_options: list[int] | None = None) -> None:
        self.budget_options = budget_options or [256, 512, 1024, 2048, 4096]

    def allocate(self, obs: Observation, turn: int, global_budget: int, used: int) -> int:
        remaining = global_budget - used
        if remaining <= 0:
            return 256

        # Early turns: moderate budget; later turns with long history: increase
        history_len = sum(len(m.get("content", "")) for m in obs.history)
        difficulty = min(1.0, history_len / 8000 + turn * 0.1)

        if "fail" in obs.env_feedback.lower() or "error" in obs.env_feedback.lower():
            difficulty = min(1.0, difficulty + 0.3)

        idx = int(difficulty * (len(self.budget_options) - 1))
        chosen = self.budget_options[idx]

        # Reserve budget for remaining estimated turns
        estimated_remaining_turns = max(1, 5 - turn)
        max_allowed = remaining // estimated_remaining_turns
        return min(chosen, max_allowed, remaining)


class LearnedTABPolicy(TABPolicy):
    """Placeholder for GRPO-trained TAB checkpoint."""

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self._fallback = HeuristicTABPolicy()

    def allocate(self, obs: Observation, turn: int, global_budget: int, used: int) -> int:
        # Phase 2: load checkpoint and run inference
        return self._fallback.allocate(obs, turn, global_budget, used)


def build_tab_policy(config: TABConfig) -> TABPolicy:
    if config.mode == "static":
        return StaticTABPolicy(budget_per_turn=config.budget_options[len(config.budget_options) // 2])
    if config.mode == "learned":
        if not config.checkpoint_path:
            raise ValueError("TAB learned mode requires checkpoint_path")
        return LearnedTABPolicy(config.checkpoint_path)
    return HeuristicTABPolicy(config.budget_options)


class TABRunner(BaseRunner):
    name = "tab"

    def __init__(
        self,
        llm,
        config: RunnerConfig | None = None,
        tab_config: TABConfig | None = None,
    ) -> None:
        super().__init__(llm, config)
        self.tab_config = tab_config or TABConfig()
        self.policy = build_tab_policy(self.tab_config)
        self._global_budget = 50_000

    def max_tokens_for_step(self, obs: Observation, turn: int) -> int:
        if turn == 0:
            self._global_budget = obs.remaining_budget
        used = self._global_budget - obs.remaining_budget
        allocated = self.policy.allocate(obs, turn, self._global_budget, used)
        return max(256, allocated)

    def should_stop_early(self, obs: Observation, turn: int, llm_text: str, step_tokens: int = 0) -> bool:
        return False
