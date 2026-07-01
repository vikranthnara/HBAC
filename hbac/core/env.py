from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from hbac.core.cost import BudgetTracker
from hbac.core.types import AgentAction, EvalResult, Observation, StepResult, TaskSpec
from hbac.training.reward import TaskControllerReward


@runtime_checkable
class AgentEnv(Protocol):
    @property
    def task_spec(self) -> TaskSpec: ...

    @property
    def remaining_budget(self) -> int: ...

    def reset(self, task_id: str) -> Observation: ...

    def step(self, action: AgentAction) -> StepResult: ...

    def evaluate(self) -> EvalResult: ...


class BaseAgentEnv(ABC):
    """Shared history, budget tracking, and reward helpers."""

    def __init__(self, budget_tokens: int, lambda_penalty: float = 0.001) -> None:
        self._budget = BudgetTracker(budget_tokens)
        self._lambda = lambda_penalty
        self._history: list[dict[str, str]] = []
        self._turn = 0
        self._task_spec: TaskSpec | None = None
        self._done = False

    @property
    def task_spec(self) -> TaskSpec:
        if self._task_spec is None:
            raise RuntimeError("Environment not reset")
        return self._task_spec

    @property
    def remaining_budget(self) -> int:
        return self._budget.remaining

    @property
    def total_tokens(self) -> int:
        return self._budget.tokens_used

    def _append_history(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})

    def _build_observation(self, env_feedback: str) -> Observation:
        return Observation(
            history=list(self._history),
            env_feedback=env_feedback,
            tools_available=self.task_spec.tools_available,
            turn=self._turn,
            remaining_budget=self.remaining_budget,
        )

    def record_llm_tokens(self, tokens: int) -> None:
        self._budget.record(tokens)

    def compute_step_reward(self, success_signal: float = 0.0) -> float:
        return success_signal - self._budget.hinge_penalty(self._lambda)

    def compute_terminal_reward(self, success: bool, **kwargs) -> float:
        reward_fn = TaskControllerReward(lambda_token=self._lambda)
        return reward_fn.terminal(
            success=success,
            tokens_used=self.total_tokens,
            budget=self._budget.budget_tokens,
            **kwargs,
        )

    @abstractmethod
    def reset(self, task_id: str) -> Observation: ...

    @abstractmethod
    def step(self, action: AgentAction) -> StepResult: ...

    @abstractmethod
    def evaluate(self) -> EvalResult: ...
