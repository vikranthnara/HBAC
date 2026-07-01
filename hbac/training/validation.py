from __future__ import annotations

from dataclasses import dataclass

from hbac.training.reward import TaskControllerReward


@dataclass
class RewardValidationResult:
    name: str
    passed: bool
    detail: str


def validate_success_dominates_premature_stop(reward: TaskControllerReward | None = None) -> RewardValidationResult:
    """Successful completion must beat premature stop at equal token cost."""
    r = reward or TaskControllerReward()
    success_reward = r.terminal(success=True, tokens_used=1000, budget=5000, env_done=True)
    hack_reward = r.terminal(
        success=False,
        tokens_used=1000,
        budget=5000,
        agent_initiated_stop=True,
        env_done=False,
    )
    passed = success_reward > hack_reward
    return RewardValidationResult(
        name="success_dominates_premature_stop",
        passed=passed,
        detail=f"success={success_reward:.4f} vs premature_stop={hack_reward:.4f}",
    )


def validate_budget_violation_penalized(reward: TaskControllerReward | None = None) -> RewardValidationResult:
    """Over-budget trajectories receive strictly lower reward than in-budget at same success."""
    r = reward or TaskControllerReward()
    in_budget = r.terminal(success=True, tokens_used=4000, budget=5000, env_done=True)
    over_budget = r.terminal(success=True, tokens_used=8000, budget=5000, env_done=True)
    passed = in_budget > over_budget
    return RewardValidationResult(
        name="budget_violation_penalized",
        passed=passed,
        detail=f"in_budget={in_budget:.4f} vs over_budget={over_budget:.4f}",
    )


def validate_token_cost_monotonic(reward: TaskControllerReward | None = None) -> RewardValidationResult:
    """Higher token use never increases reward holding success fixed."""
    r = reward or TaskControllerReward()
    low = r.terminal(success=True, tokens_used=500, budget=5000, env_done=True)
    high = r.terminal(success=True, tokens_used=3000, budget=5000, env_done=True)
    passed = low >= high
    return RewardValidationResult(
        name="token_cost_monotonic",
        passed=passed,
        detail=f"low_tokens={low:.4f} vs high_tokens={high:.4f}",
    )


def validate_no_free_stop_hack(reward: TaskControllerReward | None = None) -> RewardValidationResult:
    """Stopping immediately without success cannot beat continuing (failed) at same tokens."""
    r = reward or TaskControllerReward()
    immediate_stop = r.terminal(
        success=False,
        tokens_used=100,
        budget=5000,
        agent_initiated_stop=True,
        env_done=False,
    )
    failed_continue = r.terminal(success=False, tokens_used=100, budget=5000, env_done=True)
    passed = failed_continue >= immediate_stop
    return RewardValidationResult(
        name="no_free_stop_hack",
        passed=passed,
        detail=f"immediate_stop={immediate_stop:.4f} vs failed_continue={failed_continue:.4f}",
    )


def validate_stop_hacking_margin(
    reward: TaskControllerReward | None = None,
    min_margin: float = 0.1,
) -> RewardValidationResult:
    """Reward gap between success and premature stop must exceed min_margin."""
    r = reward or TaskControllerReward()
    success_reward = r.terminal(success=True, tokens_used=1000, budget=5000, env_done=True)
    hack_reward = r.terminal(
        success=False,
        tokens_used=1000,
        budget=5000,
        agent_initiated_stop=True,
        env_done=False,
    )
    margin = success_reward - hack_reward
    passed = margin >= min_margin
    return RewardValidationResult(
        name="stop_hacking_margin",
        passed=passed,
        detail=f"margin={margin:.4f} (min={min_margin})",
    )


def run_all_validations(reward: TaskControllerReward | None = None) -> list[RewardValidationResult]:
    return [
        validate_success_dominates_premature_stop(reward),
        validate_budget_violation_penalized(reward),
        validate_token_cost_monotonic(reward),
        validate_no_free_stop_hack(reward),
        validate_stop_hacking_margin(reward),
    ]


def sweep_reward_hyperparameters(
    lambda_values: list[float] | None = None,
    penalty_values: list[float] | None = None,
) -> list[dict]:
    """Grid search reward hyperparameters; return passing configurations."""
    lambda_values = lambda_values or [0.0005, 0.001, 0.002, 0.005]
    penalty_values = penalty_values or [0.3, 0.5, 0.7, 1.0]
    results = []
    for lam in lambda_values:
        for pen in penalty_values:
            reward = TaskControllerReward(lambda_token=lam, premature_stop_penalty=pen)
            validations = run_all_validations(reward)
            passed = all(v.passed for v in validations)
            results.append(
                {
                    "lambda_token": lam,
                    "premature_stop_penalty": pen,
                    "passed": passed,
                    "validations": [
                        {"name": v.name, "passed": v.passed, "detail": v.detail} for v in validations
                    ],
                }
            )
    return results


def all_passed(reward: TaskControllerReward | None = None) -> bool:
    return all(r.passed for r in run_all_validations(reward))


def best_reward_defaults(sweep_results: list[dict]) -> tuple[float, float]:
    """Pick smallest penalty among passing configs (parsimonious anti-hack)."""
    passing = [r for r in sweep_results if r["passed"]]
    if not passing:
        return 0.001, 0.5
    best = min(passing, key=lambda r: (r["premature_stop_penalty"], r["lambda_token"]))
    return best["lambda_token"], best["premature_stop_penalty"]
