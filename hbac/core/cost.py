from __future__ import annotations


class BudgetTracker:
    """Tracks token usage against a global per-task budget."""

    def __init__(self, budget_tokens: int) -> None:
        self.budget_tokens = budget_tokens
        self.tokens_used = 0
        self.step_tokens: list[int] = []

    def record(self, tokens: int) -> None:
        self.tokens_used += tokens
        self.step_tokens.append(tokens)

    @property
    def remaining(self) -> int:
        return max(0, self.budget_tokens - self.tokens_used)

    @property
    def violated(self) -> bool:
        return self.tokens_used > self.budget_tokens

    def hinge_penalty(self, lambda_penalty: float) -> float:
        over = max(0, self.tokens_used - self.budget_tokens)
        return lambda_penalty * over

    def can_afford(self, requested: int) -> bool:
        return self.remaining >= requested
