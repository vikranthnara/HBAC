"""Draft-model and controller overhead accounting (Phase 2 stub)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ControllerOverheadTracker:
    """Track controller inference tokens vs task budget for NetGain gate."""

    controller_tokens: int = 0
    draft_tokens: int = 0
    target_tokens: int = 0
    task_budget: int = 0

    def record_controller_step(self, tokens: int = 1) -> None:
        self.controller_tokens += tokens

    def record_draft_step(self, tokens: int) -> None:
        self.draft_tokens += tokens

    def record_target_step(self, tokens: int) -> None:
        self.target_tokens += tokens

    @property
    def overhead_fraction(self) -> float:
        total = self.controller_tokens + self.draft_tokens + self.target_tokens
        if total <= 0:
            return 0.0
        return (self.controller_tokens + self.draft_tokens) / total

    @property
    def overhead_vs_budget(self) -> float:
        if self.task_budget <= 0:
            return 0.0
        return (self.controller_tokens + self.draft_tokens) / self.task_budget


def estimate_controller_overhead(num_steps: int, tokens_per_step: int = 1) -> ControllerOverheadTracker:
    t = ControllerOverheadTracker(task_budget=50_000)
    for _ in range(num_steps):
        t.record_controller_step(tokens_per_step)
    t.record_target_step(1000)
    return t
