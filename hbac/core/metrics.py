from __future__ import annotations

import json
from pathlib import Path


class MetricsLogger:
    """Aggregate run-level metrics aligned with Research Plan §8."""

    def __init__(self) -> None:
        self.results: list[dict] = []

    def record(self, task_id: str, success: bool, total_tokens: int, budget: int, **extra) -> None:
        budget_violated = total_tokens > budget
        utility = (1.0 / total_tokens) if success and total_tokens > 0 else 0.0
        self.results.append(
            {
                "task_id": task_id,
                "success": success,
                "total_tokens": total_tokens,
                "budget": budget,
                "budget_violated": budget_violated,
                "utility_per_token": utility,
                **extra,
            }
        )

    def summarize(self) -> dict:
        if not self.results:
            return {
                "pass_at_1": 0.0,
                "budget_violation_rate": 0.0,
                "mean_tokens": 0.0,
                "mean_utility_per_token": 0.0,
                "num_tasks": 0,
            }
        n = len(self.results)
        successes = sum(1 for r in self.results if r["success"])
        violations = sum(1 for r in self.results if r["budget_violated"])
        mean_tokens = sum(r["total_tokens"] for r in self.results) / n
        utilities = [r["utility_per_token"] for r in self.results if r["success"]]
        mean_utility = sum(utilities) / len(utilities) if utilities else 0.0
        return {
            "pass_at_1": successes / n,
            "budget_violation_rate": violations / n,
            "mean_tokens": mean_tokens,
            "mean_utility_per_token": mean_utility,
            "num_tasks": n,
            "per_task": self.results,
        }

    @staticmethod
    def summarize_batch(
        task_successes: list[bool],
        task_tokens: list[int],
        task_budgets: list[int],
        global_budget: int,
        allocation_variance: float = 0.0,
    ) -> dict:
        n = len(task_successes)
        if n == 0:
            return {"pass_at_1": 0.0, "batch_violation_rate": 0.0}
        batch_used = sum(task_tokens)
        task_viol = sum(1 for t, b in zip(task_tokens, task_budgets) if t > b)
        batch_viol = int(batch_used > global_budget)
        return {
            "pass_at_1": sum(task_successes) / n,
            "batch_violation_rate": (task_viol + batch_viol) / (n + 1),
            "mean_tokens": sum(task_tokens) / n,
            "allocation_variance": allocation_variance,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summarize(), indent=2), encoding="utf-8")
