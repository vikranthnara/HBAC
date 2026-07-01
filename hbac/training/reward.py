from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Decomposed Level-2 reward terms from Research Plan §4.2.

    Grounding: R^(2) = S_i - λ C_i - γ L_i - δ R_i [A3, A4]; KL anti-hacking [A11].
    """

    success: float
    token_cost: float
    latency_cost: float
    risk_penalty: float

    @property
    def total(self) -> float:
        return self.success - self.token_cost - self.latency_cost - self.risk_penalty


@dataclass
class TaskControllerReward:
    """
    Level-2 reward: R^(2) = S_i - λ C_i - γ L_i - δ R_i

    Grounding: Re-FORC utility J = E[R*] - λT [A3]; CLEAR shadow price [A4].
    Anti-hacking: sparse terminal success + premature_stop_penalty [A11].
    """

    lambda_token: float = 0.001
    gamma_latency: float = 0.0001
    delta_risk: float = 0.01
    premature_stop_penalty: float = 0.5

    def compute(
        self,
        *,
        success: bool,
        tokens_used: int,
        budget: int,
        latency_ms: float = 0.0,
        risk_score: float = 0.0,
        stopped_early: bool = False,
        terminated_by_env: bool = False,
    ) -> RewardComponents:
        # Success only when task actually succeeded at termination
        success_val = 1.0 if success else 0.0

        # Premature stop hack: agent chose stop before env done without success
        if stopped_early and not terminated_by_env and not success:
            success_val -= self.premature_stop_penalty

        over_budget = max(0, tokens_used - budget)
        token_cost = self.lambda_token * (tokens_used + over_budget)

        latency_cost = self.gamma_latency * latency_ms
        risk_penalty = self.delta_risk * risk_score

        return RewardComponents(
            success=success_val,
            token_cost=token_cost,
            latency_cost=latency_cost,
            risk_penalty=risk_penalty,
        )

    def terminal(
        self,
        *,
        success: bool,
        tokens_used: int,
        budget: int,
        latency_ms: float = 0.0,
        risk_score: float = 0.0,
        agent_initiated_stop: bool = False,
        env_done: bool = False,
    ) -> float:
        stopped_early = agent_initiated_stop and not env_done
        return self.compute(
            success=success,
            tokens_used=tokens_used,
            budget=budget,
            latency_ms=latency_ms,
            risk_score=risk_score,
            stopped_early=stopped_early,
            terminated_by_env=env_done,
        ).total


@dataclass
class BatchReward:
    """Level-1 batch reward R_batch = sum(S_i) - lambda * sum(C_i) with global cap."""

    lambda_batch: float = 0.001

    def total(
        self,
        *,
        successes: list[bool],
        tokens: list[int],
        budgets: list[int],
        global_budget: int,
    ) -> float:
        success_val = sum(float(s) for s in successes)
        token_cost = self.lambda_batch * sum(tokens)
        over = max(0, sum(tokens) - global_budget)
        token_cost += self.lambda_batch * over
        return success_val - token_cost
