from __future__ import annotations

from pathlib import Path

from hbac.baselines.base import BaseRunner, RunnerConfig
from hbac.core.types import Observation
from hbac.training.controller import MonolithicController


class ControllerRunner(BaseRunner):
    """Eval-only runner: ReAct loop with learned stop head via should_stop_early."""

    name = "controller"

    def __init__(
        self,
        llm,
        controller: MonolithicController,
        config: RunnerConfig | None = None,
        stop_threshold: float = 0.5,
    ) -> None:
        super().__init__(llm, config)
        self.controller = controller
        self.stop_threshold = stop_threshold

    def max_tokens_for_step(self, obs: Observation, turn: int) -> int:
        return self.config.max_tokens_per_step

    def should_stop_early(
        self, obs: Observation, turn: int, llm_text: str, step_tokens: int = 0
    ) -> bool:
        return self.controller.should_stop(obs, threshold=self.stop_threshold)

    @classmethod
    def from_checkpoint(cls, llm, checkpoint: Path, **kwargs) -> ControllerRunner:
        return cls(llm, MonolithicController.load(checkpoint), **kwargs)
